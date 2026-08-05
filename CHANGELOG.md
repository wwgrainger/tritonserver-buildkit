# CHANGELOG

<!-- version list -->

## v1.11.0 (2026-08-05)

### Features

- Add cache bust parameter
  ([`1d4c082`](https://github.com/wwgrainger/tritonserver-buildkit/commit/1d4c082fd3fe2f95f7994d2e42b4d3ff9b71d7d8))


## v1.10.1 (2026-04-02)

### Bug Fixes

- Allow for relaxed float dtypes for test cases for trt compiled models
  ([`e5a4e90`](https://github.com/wwgrainger/tritonserver-buildkit/commit/e5a4e909894130032987bec9e746abfcbe26c6be))


## v1.10.0 (2026-04-01)

### Features

- Add support for instance family for selecting trt exec node
  ([`534790c`](https://github.com/wwgrainger/tritonserver-buildkit/commit/534790ca93a2ce67e213de186280c93520896fdb))


## v1.10.0-alpha.1+feature-trt-compiling (2026-04-01)

### Features

- Add support for compiling onnx models with trtexec
  ([`0da4877`](https://github.com/wwgrainger/tritonserver-buildkit/commit/0da4877d2e132dc1f6131fd4a4dd142a24da25eb))


## v1.9.0 (2026-03-31)

### Bug Fixes

- Mlflow batch size duplication
  ([`e797296`](https://github.com/wwgrainger/tritonserver-buildkit/commit/e79729621246414df857515215eac6423641daba))

- Mlflow test file shouldn't define input / output
  ([`cd7c5e3`](https://github.com/wwgrainger/tritonserver-buildkit/commit/cd7c5e3afa98d4978ab2f5e11f855f472254da66))

- Use python model
  ([`e51fba5`](https://github.com/wwgrainger/tritonserver-buildkit/commit/e51fba5a61b59b8effc1f6ed812eb5498902aa59))

### Build System

- **deps**: Bump actions/cache from 4 to 5
  ([`c12c952`](https://github.com/wwgrainger/tritonserver-buildkit/commit/c12c9520f472f65d8745ecb33db50e4d13cf11f2))

- **deps**: Bump actions/checkout from 4 to 6
  ([`7b586f3`](https://github.com/wwgrainger/tritonserver-buildkit/commit/7b586f33b2a2863c8aa2bf0a37e96803445c5bfb))

- **deps**: Bump actions/setup-python from 5 to 6
  ([`3cd201e`](https://github.com/wwgrainger/tritonserver-buildkit/commit/3cd201e20cf4d1f1e4e217e127d2b9498532b7ea))

### Features

- Update default tritonserver version to 26.03
  ([`fcb5ed1`](https://github.com/wwgrainger/tritonserver-buildkit/commit/fcb5ed1402ba77dd336092ffd90d9af5719f4edd))


## v1.8.2 (2025-11-20)

### Bug Fixes

- Improve error messaging for databricks model downloading
  ([`6318d22`](https://github.com/wwgrainger/tritonserver-buildkit/commit/6318d22c4f2fa93784b57dfadbfe5583a4d58fea))


## v1.8.1 (2025-11-18)

### Bug Fixes

- Set correct env
  ([`cea1472`](https://github.com/wwgrainger/tritonserver-buildkit/commit/cea1472ae8d6f994bf7894c9a4145432770a4965))


## v1.8.0 (2025-11-18)

### Features

- Initial public release
  ([`1200f29`](https://github.com/wwgrainger/tritonserver-buildkit/commit/1200f291833cccc2582df950ae36dbfa20603905))


## v1.7.1 (2025-11-08)

- Initial Release
