## ml
A handwritten digit classifier written from scratch in Rust using WebAssembly. The underlying model is a [one-vs-all](https://en.wikipedia.org/wiki/Multiclass_classification#One-vs.-rest) [SVM classifier](https://en.wikipedia.org/wiki/Support_vector_machine) which was trained on the MNIST dataset. The model was fitted by the [Pegasos algorithm](https://home.ttic.edu/~nati/Publications/PegasosMPB.pdf).

![image](https://github.com/user-attachments/assets/fe402450-e4cf-46f1-bfc5-0d50cdbafa4f)

## Rust part
### Build for wasm:
```bash
wasm-pack build -- --features=wasm --target=wasm32-unknown-unknown
```

### Build for training the model (native):
```
cargo run --package ml --bin ml --features=native
```

## JS part
```
npm install
npm run start
```
