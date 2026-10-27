"""Python 3.12 AST declaration/position catalog, separate from candidate inference."""

import ast
from collections import Counter

from .model import canonical, digest


class Catalog:
    """Preserve exact spans, lexical ownership and each execution-stage call attempt."""

    def __init__(self, sources, repositories, snapshot_id, include_calls=True):
        self.sources = sources
        self.repositories = repositories
        self.snapshot_id = snapshot_id
        self.nodes = []
        self.edges = []
        self.requests = []
        self.anchors = {}
        self.trees = {}
        self.byid = {}
        self.counts = Counter()
        self.include_calls = include_calls
        for repo in repositories:
            self.add_node(
                dict(
                    repository=repo["name"],
                    path="",
                    qualname=repo["name"],
                    kind="Repository",
                    occurrence="1",
                    parent=None,
                    source="",
                    start=0,
                    end=0,
                    column=0,
                    end_column=0,
                    definition_line=0,
                    name_column=0,
                    git_head=repo["head"],
                    git_status=repo["status"],
                    file_sha256="",
                    subtype="repository",
                    signature="",
                    docstring="",
                )
            )
        for source in sources:
            self.parse(source)

    def add_node(self, n):
        """Logical identity is separate from immutable snapshot identity."""
        logical = digest(
            canonical(
                [n["repository"], n["path"], n["kind"], n["qualname"], n["occurrence"]]
            )
        )
        n.update(
            logical_id="cg1:" + logical,
            node_id="cg1:" + self.snapshot_id + ":" + logical,
            snapshot_id=self.snapshot_id,
            source_sha256=digest(n["source"]),
        )
        if n["kind"] in {"Module", "Class", "Function"} and n["subtype"] != "lambda":
            n["legacy_node_id"] = (
                f"{n['repository']}:{n['path']}:{n['kind'].lower()}:{n['qualname']}#{n['occurrence']}"
            )
        self.nodes.append(n)
        self.byid[n["node_id"]] = n
        if n["parent"]:
            self.edge(
                n["parent"],
                n["node_id"],
                "CONTAINS",
                dict(provider="ast", line=n["definition_line"], column=n["column"]),
                "structural",
            )
        return n

    def edge(self, source, target, kind, evidence, status="candidate", **extra):
        """Evidence is part of identity so separate use sites never collapse."""
        e = dict(
            source=source,
            target=target,
            kind=kind,
            status=status,
            evidence=evidence,
            exhaustive=False,
            **extra,
        )
        e["edge_id"] = digest(canonical(e))
        self.edges.append(e)
        return e

    def parse(self, source):
        """Walk defaults/bases/decorators outside the deferred body, without target guesses."""
        text = source["text"]
        lines = text.splitlines(keepends=True)
        tree = ast.parse(text, type_comments=True)
        future = any(
            isinstance(n, ast.ImportFrom)
            and n.module == "__future__"
            and any(a.name == "annotations" for a in n.names)
            for n in tree.body
        )

        def col(line, offset):
            return len(lines[line - 1].encode()[:offset].decode("utf-8"))

        def span(n, full=False):
            if full:
                return "".join(lines[n["start"] - 1 : n["end"]])
            return ast.get_source_segment(text, n) or ""

        repo = next(
            n
            for n in self.nodes
            if n["kind"] == "Repository" and n["repository"] == source["repository"]
        )
        module = self.add_node(
            dict(
                repository=source["repository"],
                path=source["path"],
                qualname=source["module"].removesuffix(".__init__"),
                kind="Module",
                subtype="module",
                occurrence="1",
                parent=repo["node_id"],
                source=text,
                start=1,
                end=len(lines),
                column=0,
                end_column=len(lines[-1].rstrip("\r\n")) if lines else 0,
                definition_line=1,
                name_column=0,
                signature="",
                docstring=ast.get_docstring(tree) or "",
                file_sha256=source["file_sha256"],
                git_head=source["git_head"],
                git_status=source["git_status"],
            )
        )
        self.anchors.setdefault((source["absolute"], 1, 0), []).append(
            module["node_id"]
        )
        self.trees[module["node_id"]] = tree

        def declaration(n, owner):
            is_lambda = isinstance(n, ast.Lambda)
            kind = "Class" if isinstance(n, ast.ClassDef) else "Function"
            name = "<lambda>" if is_lambda else n.name
            self.counts[owner["node_id"], kind, name] += 1
            ordinal = self.counts[owner["node_id"], kind, name]
            display = f"<lambda>#{ordinal}" if is_lambda else name
            qual = (
                display
                if owner["kind"] == "Module"
                else owner["qualname"] + "." + display
            )
            start = n.lineno
            if not is_lambda:
                start = min([start] + [d.lineno for d in n.decorator_list])
                while start > 1:
                    prev = lines[start - 2]
                    indent = len(lines[n.lineno - 1]) - len(
                        lines[n.lineno - 1].lstrip()
                    )
                    if (
                        not prev.strip().startswith("#")
                        or len(prev) - len(prev.lstrip()) != indent
                    ):
                        break
                    start -= 1
            decorators = [] if is_lambda else [ast.unparse(x) for x in n.decorator_list]
            subtype = (
                "lambda" if is_lambda else "class" if kind == "Class" else "function"
            )
            if kind == "Function" and owner["kind"] == "Class":
                subtype = (
                    "staticmethod"
                    if "staticmethod" in decorators
                    else "classmethod"
                    if "classmethod" in decorators
                    else "method"
                )
            elif kind == "Function" and owner["kind"] == "Function" and not is_lambda:
                subtype = "nested"
            name_column = (
                col(n.lineno, n.col_offset)
                if is_lambda
                else lines[n.lineno - 1].index(
                    n.name,
                    col(n.lineno, n.col_offset)
                    + (
                        6
                        if isinstance(n, ast.AsyncFunctionDef)
                        else 4
                        if kind == "Function"
                        else 6
                    ),
                )
            )
            if kind == "Class":
                signature = f"class {n.name}({', '.join([ast.unparse(x) for x in n.bases] + [(k.arg + '=' if k.arg else '**') + ast.unparse(k.value) for k in n.keywords])})"
            elif is_lambda:
                signature = ast.unparse(n)
            else:
                signature = (
                    ("async " if isinstance(n, ast.AsyncFunctionDef) else "")
                    + f"def {n.name}({ast.unparse(n.args)})"
                    + (" -> " + ast.unparse(n.returns) if n.returns else "")
                )
            if getattr(n, "type_params", []):
                signature += (
                    " [type parameters: "
                    + ", ".join(ast.unparse(x) for x in n.type_params)
                    + "]"
                )
            node = self.add_node(
                dict(
                    repository=source["repository"],
                    path=source["path"],
                    qualname=qual,
                    kind=kind,
                    subtype=subtype,
                    occurrence=owner["occurrence"] + "." + str(ordinal),
                    parent=owner["node_id"],
                    source=span(n)
                    if is_lambda
                    else "".join(lines[start - 1 : n.end_lineno]),
                    start=start,
                    end=n.end_lineno,
                    column=col(n.lineno, n.col_offset),
                    end_column=col(n.end_lineno, n.end_col_offset),
                    definition_line=n.lineno,
                    name_column=name_column,
                    signature=signature,
                    docstring="" if is_lambda else ast.get_docstring(n) or "",
                    file_sha256=source["file_sha256"],
                    git_head=source["git_head"],
                    git_status=source["git_status"],
                    module_id=module["node_id"],
                )
            )
            self.trees[node["node_id"]] = n
            anchor = (source["absolute"], n.lineno, name_column)
            self.anchors.setdefault(anchor, []).append(node["node_id"])
            return node

        def request(
            expr,
            owner,
            phase,
            relation,
            active=True,
            role="expression",
            target_source=None,
        ):
            if not self.include_calls:
                return
            item = dict(
                source=source,
                expr=expr,
                owner_id=owner["node_id"],
                phase=phase,
                relation=relation,
                active=active,
                role=role,
                line=expr.lineno,
                column=col(expr.lineno, expr.col_offset),
                end_line=expr.end_lineno,
                end_column=col(expr.end_lineno, expr.end_col_offset),
                expression=ast.unparse(expr),
                target_source=target_source,
            )
            if relation == "CALLS":
                key = [
                    owner["node_id"],
                    item["line"],
                    item["column"],
                    item["expression"],
                    role,
                ]
                site = self.add_node(
                    dict(
                        repository=source["repository"],
                        path=source["path"],
                        qualname=owner["qualname"]
                        + ":call:"
                        + str(item["line"])
                        + ":"
                        + str(item["column"])
                        + ":"
                        + role,
                        kind="CallSite",
                        subtype=role,
                        occurrence=digest(canonical(key)),
                        parent=owner["node_id"],
                        source=span(expr),
                        start=item["line"],
                        end=expr.end_lineno,
                        column=item["column"],
                        end_column=item["end_column"],
                        definition_line=item["line"],
                        name_column=item["column"],
                        signature="",
                        docstring="",
                        file_sha256=source["file_sha256"],
                        git_head=source["git_head"],
                        git_status=source["git_status"],
                        module_id=module["node_id"],
                        execution_owner=owner["node_id"],
                        phase=phase,
                        active_call=active,
                        expression=item["expression"],
                        exhaustive=False,
                        reasons=[],
                        candidates=[],
                        rejected=[],
                        raw=[],
                    )
                )
                item["site_id"] = site["node_id"]
            self.requests.append(item)

        def annotation(n, owner, local=False, alias=False):
            if n is not None:
                visit(
                    n,
                    owner,
                    "deferred_annotation"
                    if future or local or alias
                    else "definition_annotation",
                    not (future or local or alias),
                )

        def visit(n, owner, phase="module", active=True, callee=False):
            if isinstance(
                n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
            ):
                node = declaration(n, owner)
                lam = isinstance(n, ast.Lambda)
                if not lam:
                    for deco in n.decorator_list:
                        visit(deco, owner, "definition", active)
                        request(
                            deco,
                            owner,
                            "definition",
                            "CALLS",
                            active,
                            "decorator_factory_application"
                            if isinstance(deco, ast.Call)
                            else "decorator_application",
                        )
                if isinstance(n, ast.ClassDef):
                    for base in n.bases:
                        visit(base, owner, "class_definition", active)
                        request(base, node, "class_definition", "INHERITS", active)
                    for keyword in n.keywords:
                        visit(keyword.value, owner, "class_definition", active)
                    for param in n.type_params:
                        for child in ast.iter_child_nodes(param):
                            annotation(child, owner, alias=True)
                    for statement in n.body:
                        visit(statement, node, "class_body", active)
                else:
                    for default in n.args.defaults + [
                        x for x in n.args.kw_defaults if x is not None
                    ]:
                        visit(default, owner, "definition", active)
                    if not lam:
                        for arg in (
                            n.args.posonlyargs
                            + n.args.args
                            + n.args.kwonlyargs
                            + ([n.args.vararg] if n.args.vararg else [])
                            + ([n.args.kwarg] if n.args.kwarg else [])
                        ):
                            annotation(arg.annotation, owner)
                        annotation(n.returns, owner)
                        for param in n.type_params:
                            for child in ast.iter_child_nodes(param):
                                annotation(child, owner, alias=True)
                    if lam:
                        visit(n.body, node, "lambda_body", active)
                    else:
                        for statement in n.body:
                            visit(statement, node, "function_body", True)
                return
            if isinstance(n, ast.AnnAssign):
                visit(n.target, owner, phase, active)
                if n.value:
                    visit(n.value, owner, phase, active)
                annotation(n.annotation, owner, local=owner["kind"] == "Function")
                return
            if isinstance(n, ast.TypeAlias):
                annotation(n.value, owner, alias=True)
                for param in n.type_params:
                    for child in ast.iter_child_nodes(param):
                        annotation(child, owner, alias=True)
                return
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                for alias in n.names:
                    if alias.name == "*":
                        continue
                    request(alias, owner, phase, "IMPORTS", active)
                return
            if isinstance(n, ast.Call):
                request(n.func, owner, phase, "CALLS", active)
                visit(n.func, owner, phase, active, callee=True)
                for arg in n.args:
                    visit(arg, owner, phase, active)
                for kw in n.keywords:
                    visit(kw.value, owner, phase, active)
                return
            if (
                isinstance(n, (ast.Name, ast.Attribute))
                and isinstance(n.ctx, ast.Load)
                and not callee
            ):
                request(n, owner, phase, "REFERENCES", active)
            for child in ast.iter_child_nodes(n):
                visit(child, owner, phase, active)

        for statement in tree.body:
            visit(statement, module)
