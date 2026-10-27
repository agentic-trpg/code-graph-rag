"""Bounded Python callable values with branch joins, overwrite kills and unknown origins.

This Python-only profile owns its source catalog and never promotes liveness
references to calls. It is not a Python interpreter or a whole-program soundness
claim. Unsupported values retain explicit unknown provenance.
"""

import ast
from collections import defaultdict

UNKNOWN = frozenset({("unknown", "source-not-enumerated")})


def union(*values):
    return frozenset().union(*values)


class Flow:
    """Use fresh catalog ASTs; keep references, call attempts and value origins separate."""

    def __init__(self, catalog):
        self.c = catalog
        self.nodes = catalog.byid
        self.env = defaultdict(dict)
        self.params = defaultdict(lambda: UNKNOWN)
        self.returns = defaultdict(frozenset)
        self.maps = defaultdict(dict)
        self.bases = defaultdict(set)
        self.result = {}
        self.ast_nodes = {id(t): self.nodes[i] for i, t in catalog.trees.items()}
        self.modules = {
            s["module"].removesuffix(".__init__"): next(
                n
                for n in catalog.nodes
                if n["kind"] == "Module"
                and (n["repository"], n["path"]) == (s["repository"], s["path"])
            )
            for s in catalog.sources
        }
        self.sources = {(s["repository"], s["path"]): s for s in catalog.sources}

    def lookup(self, name, owner, env):
        if name in env:
            return env[name]
        current = self.nodes[owner["parent"]] if owner.get("parent") else None
        while current and current["kind"] != "Repository":
            # Class namespaces do not enclose method bodies (Python LEGB).
            if current["kind"] != "Class" and name in self.env[current["node_id"]]:
                return self.env[current["node_id"]][name]
            current = self.nodes.get(current.get("parent"))
        return UNKNOWN

    def attr(self, values, name, seen=None):
        found = set()
        seen = seen or set()
        for tag, ident in values:
            if tag in {"class", "instance", "module", "super"}:
                if ident in seen:
                    continue
                seen.add(ident)
                owners = self.bases[ident] if tag == "super" else {ident}
                for parent in owners:
                    members = [
                        n
                        for n in self.c.nodes
                        if n["parent"] == parent
                        and n["kind"] in {"Function", "Class"}
                        and n["qualname"].rsplit(".", 1)[-1] == name
                    ]
                    if members:
                        found.update(
                            (
                                "function" if n["kind"] == "Function" else "class",
                                n["node_id"],
                            )
                            for n in members
                        )
                    elif name in self.env[parent]:
                        found.update(self.env[parent][name])
                    else:
                        found.update(
                            self.attr(
                                frozenset(("class", b) for b in self.bases[parent]),
                                name,
                                seen,
                            )
                        )
            elif tag == "external":
                found.add(("external", ident + "." + name))
            elif tag == "unknown":
                found.add((tag, ident))
        return frozenset(found) or UNKNOWN

    def position(self, n, owner, role="expression"):
        s = self.sources[owner["repository"], owner["path"]]
        line = s["text"].splitlines()[n.lineno - 1]
        col = len(line.encode()[: n.col_offset].decode())
        return (s["absolute"], n.lineno, col, ast.unparse(n), role)

    def observe(self, n, owner, values, role="expression"):
        key = self.position(n, owner, role)
        prior = self.result.get(key, frozenset())
        self.result[key] = union(prior, values)

    def annotation(self, node, owner):
        """Treat declared return types as candidates, preserving unknown runtime origins."""
        if node is None:
            return frozenset()
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                return self.annotation(ast.parse(node.value, mode="eval").body, owner)
            except SyntaxError:
                return frozenset()
        if isinstance(node, ast.Name):
            parent = self.nodes.get(owner.get("parent"))
            if node.id == "Self" and parent and parent["kind"] == "Class":
                return frozenset({("instance", parent["node_id"])})
            values = self.lookup(node.id, owner, {})
            return frozenset(
                ("instance", ident) for tag, ident in values if tag == "class"
            )
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return union(
                self.annotation(node.left, owner), self.annotation(node.right, owner)
            )
        if isinstance(node, ast.Subscript):
            return self.annotation(node.slice, owner)
        if isinstance(node, ast.Tuple):
            return union(*(self.annotation(x, owner) for x in node.elts))
        if isinstance(node, ast.Attribute):
            value = self.attr(self.evaluate(node.value, owner, {}, True), node.attr)
            return frozenset(
                ("instance", ident) for tag, ident in value if tag == "class"
            )
        return frozenset()

    def evaluate(self, n, owner, env, callee=False):
        if n is None:
            return frozenset()
        if isinstance(n, ast.Name):
            value = self.lookup(n.id, owner, env)
            if not callee:
                self.observe(n, owner, value)
            return value
        if isinstance(n, ast.Constant):
            return frozenset({("constant", repr(n.value))})
        if isinstance(n, ast.Lambda):
            return frozenset({("function", self.ast_nodes[id(n)]["node_id"])})
        if isinstance(n, ast.Attribute):
            value = self.attr(self.evaluate(n.value, owner, env), n.attr)
            if not callee:
                self.observe(n, owner, value)
            return value
        if isinstance(n, ast.Dict):
            ident = owner["node_id"] + ":" + str(n.lineno) + ":" + str(n.col_offset)
            for k, v in zip(n.keys, n.values):
                key = repr(k.value) if isinstance(k, ast.Constant) else "*"
                self.maps[ident][key] = self.evaluate(v, owner, env)
            return frozenset({("map", ident)})
        if isinstance(n, ast.Subscript):
            values = self.evaluate(n.value, owner, env)
            fixed = repr(n.slice.value) if isinstance(n.slice, ast.Constant) else None
            out = []
            for tag, ident in values:
                if tag == "map":
                    out.extend(
                        [
                            self.maps[ident].get(fixed, frozenset()),
                            self.maps[ident].get("*", frozenset()),
                        ]
                        if fixed is not None
                        else list(self.maps[ident].values())
                    )
                elif tag == "unknown":
                    out.append(UNKNOWN)
            return union(*out) if out else UNKNOWN
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name) and n.func.id == "super":
                parent = self.nodes.get(owner.get("parent"))
                return (
                    frozenset({("super", parent["node_id"])})
                    if parent and parent["kind"] == "Class"
                    else UNKNOWN
                )
            values = self.evaluate(n.func, owner, env, True)
            self.observe(n.func, owner, values)
            arguments = [self.evaluate(x, owner, env) for x in n.args]
            keywords = {
                k.arg: self.evaluate(k.value, owner, env) for k in n.keywords if k.arg
            }
            returned = []
            for tag, ident in values:
                if tag == "class":
                    returned.append(frozenset({("instance", ident)}))
                    continue
                if tag != "function":
                    continue
                target = self.nodes[ident]
                tree = self.c.trees[ident]
                params = tree.args.posonlyargs + tree.args.args
                receiver = (
                    self.evaluate(n.func.value, owner, env)
                    if isinstance(n.func, ast.Attribute)
                    else frozenset()
                )
                if target["subtype"] == "classmethod" or (
                    target["subtype"] == "method"
                    and any(t in {"instance", "super"} for t, _ in receiver)
                ):
                    params = params[1:]
                for param, value in zip(params, arguments):
                    self.params[ident, param.arg] = union(
                        self.params[ident, param.arg], value
                    )
                for key, value in keywords.items():
                    self.params[ident, key] = union(self.params[ident, key], value)
                returned.append(
                    union(
                        self.returns[ident],
                        self.annotation(getattr(tree, "returns", None), target),
                    )
                )
            return union(*returned) if returned else UNKNOWN
        if isinstance(n, ast.IfExp):
            return union(
                self.evaluate(n.body, owner, env), self.evaluate(n.orelse, owner, env)
            )
        if isinstance(n, (ast.Tuple, ast.List, ast.Set)):
            return union(*(self.evaluate(x, owner, env) for x in n.elts))
        if isinstance(n, ast.BoolOp):
            return union(*(self.evaluate(x, owner, env) for x in n.values))
        for child in ast.iter_child_nodes(n):
            self.evaluate(child, owner, env)
        return UNKNOWN

    def bind(self, target, value, owner, env):
        if isinstance(target, ast.Name):
            env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                self.bind(item, union(value, UNKNOWN), owner, env)
        elif isinstance(target, ast.Subscript):
            keys = self.evaluate(target.slice, owner, env)
            known = [v for t, v in keys if t == "constant"]
            if any(t == "unknown" for t, _ in keys):
                known.append("*")
            for tag, ident in self.evaluate(target.value, owner, env):
                if tag == "map":
                    for key in known:
                        self.maps[ident][key] = value

    def statements(self, statements, owner, env):
        for n in statements:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                child = self.ast_nodes[id(n)]
                env[n.name] = frozenset(
                    {
                        (
                            "class" if isinstance(n, ast.ClassDef) else "function",
                            child["node_id"],
                        )
                    }
                )
                for deco in n.decorator_list:
                    value = self.evaluate(deco, owner, env)
                    self.observe(
                        deco,
                        owner,
                        value,
                        "decorator_factory_application"
                        if isinstance(deco, ast.Call)
                        else "decorator_application",
                    )
                if isinstance(n, ast.ClassDef):
                    for base in n.bases:
                        value = self.evaluate(base, owner, env)
                        self.observe(base, child, value)
                        self.bases[child["node_id"]].update(
                            i for t, i in value if t == "class"
                        )
                    for k in n.keywords:
                        self.evaluate(k.value, owner, env)
                    inner = {}
                    self.statements(n.body, child, inner)
                    self.env[child["node_id"]] = inner
                else:
                    for default in n.args.defaults + [
                        x for x in n.args.kw_defaults if x is not None
                    ]:
                        self.evaluate(default, owner, env)
                continue
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                current = next(
                    s["module"].removesuffix(".__init__")
                    for s in self.c.sources
                    if (s["repository"], s["path"])
                    == (owner["repository"], owner["path"])
                )
                prefix = (n.module or "") if isinstance(n, ast.ImportFrom) else ""
                if isinstance(n, ast.ImportFrom) and n.level:
                    package = (
                        current
                        if owner["path"].endswith("__init__.py")
                        else current.rpartition(".")[0]
                    )
                    parts = package.split(".")
                    prefix = ".".join(
                        parts[: len(parts) - n.level + 1] + ([prefix] if prefix else [])
                    )
                for alias in n.names:
                    if isinstance(n, ast.Import):
                        module = self.modules.get(alias.name)
                        value = (
                            frozenset({("module", module["node_id"])})
                            if module
                            else frozenset({("external", alias.name)})
                        )
                    else:
                        module = self.modules.get(prefix)
                        value = (
                            self.attr(
                                frozenset({("module", module["node_id"])}), alias.name
                            )
                            if module
                            else frozenset({("external", prefix + "." + alias.name)})
                        )
                    env[alias.asname or alias.name.split(".")[0]] = value
                    self.observe(alias, owner, value)
                continue
            if isinstance(n, ast.If):
                self.evaluate(n.test, owner, env)
                left = dict(env)
                right = dict(env)
                self.statements(n.body, owner, left)
                self.statements(n.orelse, owner, right)
                for name in left.keys() | right.keys():
                    env[name] = union(left.get(name, UNKNOWN), right.get(name, UNKNOWN))
                continue
            if isinstance(n, (ast.Assign, ast.AnnAssign)):
                value = self.evaluate(n.value, owner, env)
                for target in n.targets if isinstance(n, ast.Assign) else [n.target]:
                    self.bind(target, value, owner, env)
                continue
            if isinstance(n, ast.Return):
                self.returns[owner["node_id"]] = union(
                    self.returns[owner["node_id"]], self.evaluate(n.value, owner, env)
                )
                continue
            if isinstance(n, (ast.For, ast.While, ast.Try)):
                incoming = dict(env)
                for child in ast.iter_child_nodes(n):
                    if isinstance(child, ast.stmt):
                        self.statements([child], owner, env)
                    else:
                        self.evaluate(child, owner, env)
                for name in incoming.keys() | env.keys():
                    env[name] = union(
                        incoming.get(name, UNKNOWN), env.get(name, UNKNOWN)
                    )
                continue
            for child in ast.iter_child_nodes(n):
                self.evaluate(child, owner, env)

    def run(self, rounds=8):
        for _ in range(rounds):
            before = (repr(self.params), repr(self.returns), repr(self.maps))
            for node in self.c.nodes:
                if node["kind"] == "Module":
                    env = {}
                    self.statements(self.c.trees[node["node_id"]].body, node, env)
                    self.env[node["node_id"]] = env
            for node in self.c.nodes:
                if node["kind"] != "Function":
                    continue
                tree = self.c.trees[node["node_id"]]
                args = tree.args.posonlyargs + tree.args.args + tree.args.kwonlyargs
                env = {a.arg: self.params[node["node_id"], a.arg] for a in args}
                parent = self.nodes[node["parent"]]
                if (
                    args
                    and parent["kind"] == "Class"
                    and node["subtype"] != "staticmethod"
                ):
                    env[args[0].arg] = frozenset(
                        {
                            (
                                "class"
                                if node["subtype"] == "classmethod"
                                else "instance",
                                parent["node_id"],
                            )
                        }
                    )
                if isinstance(tree, ast.Lambda):
                    self.returns[node["node_id"]] = union(
                        self.returns[node["node_id"]],
                        self.evaluate(tree.body, node, env),
                    )
                else:
                    self.statements(tree.body, node, env)
                self.env[node["node_id"]] = env
            if before == (repr(self.params), repr(self.returns), repr(self.maps)):
                break
        return self.result
