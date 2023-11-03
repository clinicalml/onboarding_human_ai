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
//document.getElementById("instructions").style.display = "none";
//document.getElementById("test_timer_button").style.display = "none";
// Timed button to read instructions

var timedButton = document.getElementById("test_timer_button");
let time = 1;
let timer = setInterval(function () {
  if (time > 0) {
    time--;
    timedButton.disabled = true;
    timedButton.innerHTML =
      "Please read instructions  (" + time + " secs left)";
    console.log(time);
  }
  if (time === 0) {
    timedButton.disabled = false;
    timedButton.innerHTML = "Begin Experiment";
    // clearInterval(timer);
  }
}, 1000);


var show_instr_button = document.getElementById("show_instr_button");

show_instr_button.onclick = function () {
    // check if it was clicked 
    // if it was clicked, show instructions
    var current_display = document.getElementById("instructions").style.display;
    if (current_display == "none") {
        document.getElementById("instructions").style.display = "block";
        show_instr_button.innerHTML = "Hide instructions";
    } else {
        document.getElementById("instructions").style.display = "none";
        show_instr_button.innerHTML = "Show instructions";
    }
}


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
var ai_answers;
var ai_images;
var ai_answers_raw;
var ai_hidden = 0;

var submitButton = document.getElementById("submitButton");
var AIButton = document.getElementById("AIButton");


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

AIButton.onclick = function () {
  selection = "ai";
};

submitButton.onclick = function () {
  // check if one of the buttons is clicked
  if (
    oneButton.disabled == true &&
    twoButton.disabled == true &&
    threeButton.disabled == true &&
    fourButton.disabled == true &&
    AIButton.disabled == true
  ){     alert("Please choose an answer");
    selection = undefined;
    } 
else {
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
      getStorageImage(images[count], ai_images[count]);
      loadAiAnswer(ai_answers[count], ai_answers_raw[count]);
      updateProgressBar();
    } else {
      // go to end task page
      disableBeforeUnload();
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
      testing_answers_withai: personAnswers,
      testing_times_withai: test_pressed_times,
    })
    .then(() => {
      console.log("Document successfully written!");
    })
    .catch((error) => {
      console.error("Error writing document: ", error);
    });
}

function getStorageImage(image_name, ai_image_name) {
  document.getElementById("question_box").innerHTML = image_name;   

}

function loadAiAnswer(ai_answer, ai_answer_raw) {
  var elem = document.getElementById("ai_answer");
  elem.innerHTML = ai_answer;
  
}

function loadTaskData() {
  db.collection("tasks_mmlu")
    .doc(task_id)
    .get()
    .then(function (query_snapshot) {
      rand_task = query_snapshot;
      images = query_snapshot.data().testing_withai_images;
      ai_images = query_snapshot.data().testing_withai_images_box;
      ai_answers = query_snapshot.data().testing_withai_ai_answers;
      ai_answers_raw = query_snapshot.data().testing_withai_ai_answers_raw;
      getStorageImage(images[0], ai_images[0]);
      loadAiAnswer(ai_answers[0], ai_answers_raw[0]);
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
// enableBeforeUnload();
