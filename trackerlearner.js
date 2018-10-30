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

const IDLE_LEARNING_DELAY = 10;
const MIN_CONFIDENCE = 0.25;
const EPOCHS = 100;
const VALIDATION_SPLIT = 0.2;

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

  setTimeout(() => continueLearning(), 1000 * IDLE_LEARNING_DELAY);
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

  await model.fit(xs, ys, { epochs: 20, validationSplit: 0.1, callbacks });

  await model.save("localstorage://page-history-model");

  console.info("Applied ongoing learning")
}

async function learn() {
  console.info("Building Model");

  const button = document.getElementById("learn");
  button.textContent = `Learning... Constructing Deep Neural Net`;
  button.disabled = true;

  const model = tf.sequential();

  model.add(tf.layers.dense({ units: 15, activation: "relu", inputShape: [5, PAGES.length] }));
  model.add(tf.layers.dense({ units: 20, activation: "relu" }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: PAGES.length, activation: "softmax" }));

  model.compile({ loss: "categoricalCrossentropy", optimizer: "adam" });

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

  const fitData = await model.fit(xs, ys, { epochs: EPOCHS, validationSplit: VALIDATION_SPLIT, callbacks });

  xs.dispose();
  ys.dispose();

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
      responsive: true
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

  const prediction = model.predict(tf.tidy(() => tf.stack([tf.oneHot(last5, PAGES.length)])));

  const predictionData = await prediction.data();

  printChart(predictionData);
  showQuickActions(predictionData);
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
      responsive: true,
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

function showQuickActions(predictionData) {

  const predictions = Array.from(predictionData)
    .map((confidence,i) => ({ page: PAGES[i], confidence }))
    .filter(prediction => prediction.confidence >= MIN_CONFIDENCE);

  if (predictions.length === 0) {
    return;
  } 

  predictions.sort((a,b) => b.confidence - a.confidence);
  
  const quickActionLinks = document.getElementById("quick-action-links");

  for (let prediction of predictions) {
    const link = document.createElement("A");
    link.href = "index.html?page=" + encodeURIComponent(prediction.page);
    link.className = "card-link";
    link.textContent = `${prediction.page} (${Math.round(prediction.confidence * 100)}% confident)`;
    
    const li = document.createElement("LI");
    li.className = "list-group-item";
    li.appendChild(link);
    
    quickActionLinks.appendChild(li);
  }

  document.getElementById("quick-actions").classList.add("in");
}