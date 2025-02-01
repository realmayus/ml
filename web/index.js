import * as wasm from "ml";

wasm.load_model();

// allow user to paint a digit on the canvas with id="canvas"
const canvas = document.getElementById("canvas");
canvas.width = 500;
canvas.height = 500;
const ctx = canvas.getContext("2d");
ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.lineWidth = 20;
ctx.lineCap = "round";
ctx.strokeStyle = "white";
let painting = false;
let lastX = 0;
let lastY = 0;

canvas.addEventListener("mousedown", (e) => {
  painting = true;
  [lastX, lastY] = [e.offsetX, e.offsetY];
});

canvas.addEventListener("mousemove", (e) => {
    if (!painting) return;
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    [lastX, lastY] = [e.offsetX, e.offsetY];
});

const chart = document.getElementById('myChart');
let prediction = '';
let confidences = {};
const confidencesChart = new Chart(chart, {
    type: 'bar',
    data: {
        labels: Object.keys(confidences),
        datasets: [{
            label: 'Confidence Score',
            data: Object.values(confidences),
            backgroundColor: 'rgba(75, 192, 192, 0.5)',
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1
        }]
    },
    options: {
        plugins: {
            title: {
                display: true,
                text: `SVM Confidence Scores (Prediction: ${prediction})`
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                title: {
                    display: true,
                    text: 'Confidence'
                }
            },
            x: {
                title: {
                    display: true,
                    text: 'SVM Label'
                }
            }
        }
    }
});


canvas.addEventListener("mouseup", () => {
    painting = false;
    const image = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const pixels = new Uint8Array(image.data);
    const output = wasm.predict(pixels);
    let {prediction, confidences} = JSON.parse(output);
    confidences = Object.values(confidences).sort((a, b) => a[0] - b[0]);
    confidencesChart.width = 500;
    confidencesChart.data.labels = Object.values(confidences).map((x) => x[0]);
    confidencesChart.data.datasets[0].data = Object.values(confidences).map((x) => x[1]);
    confidencesChart.options.plugins.title.text = `SVM Confidence Scores (Prediction: ${prediction})`;
    confidencesChart.update();
});
