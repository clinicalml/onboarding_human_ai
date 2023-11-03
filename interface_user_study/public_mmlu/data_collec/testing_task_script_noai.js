// Import the functions you need from the SDKs you need
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// Your web app's Firebase configuration
var firebaseConfig = {

};
// Initialize Firebase
firebase.initializeApp(firebaseConfig);
firebase.analytics();

var db = firebase.firestore();
var storage = firebase.storage();
var response_id;
var task_id;
var treatment;

document.getElementById("tasks").style.display = "none";

// Timed button to read instructions
var timedButton = document.getElementById("test_timer_button");
let time = 20;
let timer = setInterval(function () {
  if (time > 0) {
    time--;
    timedButton.disabled = true;
    timedButton.innerHTML =
      "Please read instructions first (" + time + " secs left)";
    console.log(time);
  }
  if (time === 0) {
    timedButton.disabled = false;
    timedButton.innerHTML = "Begin Experiment";
    // clearInterval(timer);
  }
}, 1000); // what does 1000 do? TODO

// When timed button is clicked, show first task
timedButton.onclick = function () {
  document.getElementById("instructions").style.display = "none";
  document.getElementById("tasks").style.display = "block"; // show first task
  console.log(time);
  document.getElementById("test_timer_button").style.display = "none";
  test_pressed_times.push(new Date().getTime());
  return false; // stops page from reloading
};

// Teaching task: flips through images, asks user to select answer to question
let count = 0;
let images;
let personAnswers = [];
var test_pressed_times = [];


var oneButton = document.getElementById("oneButton");
var twoButton = document.getElementById("twoButton");
var threeButton = document.getElementById("threeButton");
var fourButton = document.getElementById("fourButton");

var submitButton = document.getElementById("submitButton");

let selection;
oneButton.onclick = function () {
  selection = 1;

};
twoButton.onclick = function () {
  selection = 2;

};
threeButton.onclick = function () {
  selection = 3;

};
fourButton.onclick = function () {
  selection = 4;

};

submitButton.onclick = function () {
  //   console.log("push", selection, count, answers[count]);
  if (selection == undefined) {
    alert("Please choose an answer");
  } else {
    personAnswers.push(selection);
    test_pressed_times.push(new Date().getTime());
    writeUserData();
    selection = undefined;
    if (count < images.length - 1) {
      count += 1;
      console.log(selection);
      getStorageImage(images[count]);
      updateProgressBar();
    } else {
      // go to end task page
      disableBeforeUnload();

      if (treatment == 1) {
        window.location.href = "./testing_task_ai.html";
      } else {
        window.location.href = "../end_task.html";
      }
    }
  }
};

function showlocalstorage() {
  console.log(response_id);
  console.log(task_id);
}
function loadlocalstorage() {
  var myData = localStorage["objectToPass"];
  myData = JSON.parse(myData);
  response_id = myData[0];
  task_id = myData[1];
  treatment = myData[3];
  //showlocalstorage();
}

function writeUserData() {
  db.collection("responses_mmlu")
    .doc(response_id)
    .update({
      testing_answers_noai: personAnswers,
      testing_times_noai: test_pressed_times,
    })
    .then(() => {
      console.log("Document successfully written!");
    })
    .catch((error) => {
      console.error("Error writing document: ", error);
    });
}

function getStorageImage(image_name) {
  document.getElementById("question_box").innerHTML = image_name;   

}

function loadTaskData() {
  db.collection("tasks_mmlu")
    .doc(task_id)
    .get()
    .then(function (query_snapshot) {
      rand_task = query_snapshot;
      images = query_snapshot.data().testing_withairec_images;
      getStorageImage(images[0]);
    })
    .catch(function (error) {
      console.log("Error getting documents: ", error);
    });
}

function updateProgressBar() {
  var elem = document.getElementById("prog_bar");
  var width = (count / images.length) * 100;
  elem.style.width = width + "%";
  elem.innerHTML = count + "/" + images.length;
}

function enableBeforeUnload() {
  window.onbeforeunload = function (e) {
    return "Discard changes?";
  };
}
function disableBeforeUnload() {
  window.onbeforeunload = null;
}

loadlocalstorage();
loadTaskData();
enableBeforeUnload();
