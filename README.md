## ml

### Build for wasm:
```bash
wasm-pack build -- --features=wasm --target=wasm32-unknown-unknown
```

### Build for training the model (native):
```
cargo run --package ml --bin ml --features=native
```