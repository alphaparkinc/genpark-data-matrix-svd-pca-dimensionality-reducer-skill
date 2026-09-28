from client import DataMatrixSvdPcaDimensionalityReducer
import json

def main():
    reducer = DataMatrixSvdPcaDimensionalityReducer()
    res = reducer.run_benchmark_pca_reducer()
    print("PCA Dimensionality Reducer Benchmark Result:")
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
