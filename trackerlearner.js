const PAGES = [
  "home",  
  "content", 
  "seo",
  "leads", 
  "reports", 
  "assets",
  "users",
  "settings",
];

const IDLE_SECONDS = 10;
const MIN_CERTAINTY = 0.5;

async function init() {
  const pageHistory = JSON.parse(localStorage.getItem("pages_visited") || "[]");

  const params = new URLSearchParams(location.search);

  const pageIndex = PAGES.indexOf(params.get("page") || "home");

  pageHistory.push(pageIndex);

  localStorage.setItem("pages_visited", JSON.stringify(pageHistory));

  console.info(`Appended page "${pageIndex}". Length = ${pageHistory.length}`);

  if (pageHistory.length > 6) {
    await predict(pageHistory);
  }

  setInterval(() => continueLearning(), 1000 * IDLE_SECONDS);
}

async function continueLearning() {
  const model = await tf.loadModel("localstorage://page-history-model");

  model.compile({ loss: "categoricalCrossentropy", optimizer: "sgd" });

  const pageHistory = JSON.parse(localStorage.getItem("pages_visited") || "[]");
  
  const { xs, ys } = getTrainingData(pageHistory);

  const callbacks = {
    onEpochEnd(epoch,data) {
      console.info(`Continuing Training: Epoch=${epoch} Loss=${data.loss.toFixed(3)}`);
    }
  };

  await model.fit(xs, ys, { epochs: 10, validationSplit: 0.1, callbacks });

  await model.save("localstorage://page-history-model");

  console.info("Applied ongoing learning")
}

async function learn() {
  console.info("Building Model");

  const button = document.getElementById("learn");
  button.textContent = `Learning... Constructing Deep Neural Net`;
  button.disabled = true;

  const model = tf.sequential();
  
  model.add(tf.layers.dense({ units: 10, activation: "relu", inputShape: [5,PAGES.length]}));
  model.add(tf.layers.dense({ units: 15, activation: "relu" }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: PAGES.length, activation: "softmax" }));

  model.compile({ loss: "categoricalCrossentropy", optimizer: "sgd" });

  console.info("Model Compiled");

  const pageHistory = JSON.parse(localStorage.getItem("pages_visited") || "[]");

  if (pageHistory.length <= 6) {
    alert("Must have at least 6 pages in history");
    return;
  }

  const callbacks = {
    onEpochEnd(epoch,data) {
      button.textContent = `Learning... Epoch: ${epoch} Loss: ${data.loss.toFixed(3)}`;
      console.info(`Training: Epoch=${epoch} Loss=${data.loss.toFixed(3)}`);
    }
  };

  const { xs, ys } = getTrainingData(pageHistory);

  const fitData = await model.fit(xs, ys, { epochs: 300, validationSplit: 0.1, callbacks });

  console.info("Model fit complete.");

  await model.save("localstorage://page-history-model");

  console.info("Model saved");

  new Chart(document.getElementById("learning-graph").getContext("2d"), {
    type: "scatter",
    data: {
      datasets: [{
        label: "Loss",
        data: fitData.history.loss.map((y,x)=>({x,y})),
      }]
    },
    options: {
      responsive: false
    }
  });
}

function getTrainingData(pageHistory) {
  return tf.tidy(() => {
    const trainingData = [];
    const trainingLabels = [];

    for (let i = 5; i < pageHistory.length; i++) {
      const last5 = [
        pageHistory[i-5],
        pageHistory[i-4],
        pageHistory[i-3],
        pageHistory[i-2],
        pageHistory[i-1]
      ];

      trainingData.push(last5);
      trainingLabels.push(pageHistory[i]);
    }

    const xs = tf.stack(trainingData.map(last5 => tf.oneHot(last5, PAGES.length)));
    const ys = tf.oneHot(trainingLabels, PAGES.length);
    
    return { xs, ys };
  });
} 

function reset() {
  localStorage.clear();
}

async function predict(pageHistory) {
  if (pageHistory.length <= 6) {
    alert("Must have at least 6 pages in history");
    return;
  }

  const last5 = [
    pageHistory[pageHistory.length-5],
    pageHistory[pageHistory.length-4],
    pageHistory[pageHistory.length-3],
    pageHistory[pageHistory.length-2],
    pageHistory[pageHistory.length-1]
  ];

  const model = await tf.loadModel("localstorage://page-history-model");

  console.info("Loaded ML model");

  const prediction = model.predict(tf.stack([tf.oneHot(last5, PAGES.length)]));

  const predictionData = await prediction.data();

  printChart(predictionData);
  highlightLinks(predictionData);
}

function printChart(predictionData) {
  new Chart(document.getElementById("predictions-graph").getContext("2d"), {
    type: "bar",
    data: {
      labels: PAGES,
      datasets: [{
        label: "Confidence",
        data: predictionData
      }]
    },
    options: {
      responsive: false,
      scales: {
        yAxes: [{
            ticks: {
                beginAtZero:true
            }
        }]
      }
    }
  });
}

function highlightLinks(predictionData) {
  let maxValue = Number.MIN_VALUE;
  let maxIndex = -1;
  for (let i = 0; i < predictionData.length; i++) {
    if (predictionData[i] > maxValue) {
      maxValue = predictionData[i];
      maxIndex = i;
    }
  }

  if (maxValue < MIN_CERTAINTY) {
    return;
  }

  const pageName = PAGES[maxIndex];

  const links = document.querySelectorAll(`a[href='index.html?page=${pageName}']`);

  for (let link of links) {
    link.classList.add("predicted-next");
  }
}

function highlightActiveLinks() {
  const links = document.querySelectorAll(`a[href='${location.href}']`);

  for (let link of links) {
    link.classList.add("current-item");
  }
}