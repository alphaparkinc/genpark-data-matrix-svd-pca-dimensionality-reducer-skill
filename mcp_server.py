import sys, json
from client import DataMatrixSvdPcaDimensionalityReducer

def main():
    engine = DataMatrixSvdPcaDimensionalityReducer()
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        try:
            req = json.loads(line)
            method = req.get("method")
            rid = req.get("id")
            params = req.get("params", {})

            if method == "tools/list":
                res = {
                    "tools": [
                        {"name": "fit_transform", "description": "Reduce dimension using PCA.", "inputSchema": {"type": "object", "properties": {"X": {"type": "array"}, "n_components": {"type": "integer"}}, "required": ["X"]}},
                        {"name": "run_benchmark_pca_reducer", "description": "Run self-test.", "inputSchema": {"type": "object"}}
                    ]
                }
            elif method == "tools/call":
                tname = params.get("name")
                args = params.get("arguments", {})
                if tname == "fit_transform":
                    out = engine.fit_transform(args.get("X", []), int(args.get("n_components", 2)))
                elif tname == "run_benchmark_pca_reducer":
                    out = engine.run_benchmark_pca_reducer()
                else:
                    out = {"error": f"Unknown tool {tname}"}
                res = {"content": [{"type": "text", "text": json.dumps(out)}]}
            else:
                res = {"error": "Unsupported method"}
            print(json.dumps({"jsonrpc": "2.0", "id": rid, "result": res}), flush=True)
        except Exception as e:
            print(json.dumps({"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e)}}), flush=True)

if __name__ == "__main__":
    main()
