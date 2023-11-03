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
document.getElementById("feedback_answer").style.display = "none";

//document.getElementById("instructions").style.display = "none";
//document.getElementById("test_timer_button").style.display = "none";
// Timed button to read instructions

var timedButton = document.getElementById("test_timer_button");
let time = 20;
let timer = setInterval(function () {
  if (time > 0) {
    time--;
    timedButton.disabled = true;
    timedButton.innerHTML =
      "Please read instructions  (" + time + " secs left)";
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
};

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
var labels_raw;
var ai_answers_raw;
var ai_hidden = 0;
var AIButton = document.getElementById("AIButton");
var nextButton = document.getElementById("nextButton");



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
  ) {
    alert("Please choose an answer");
    selection = undefined;
  } else {
    //   console.log("push", selection, count, answers[count]);
    if (selection == undefined) {
      alert("Please choose an answer");
    } else {
      personAnswers.push(selection);
      test_pressed_times.push(new Date().getTime());
      writeUserData();
      selection = undefined;
      // get div prediction_interface
      // set display to none
      // get feedback_answer
      // set display to block
      var prediction_interface = document.getElementById(
        "prediction_interface"
      );
      prediction_interface.style.display = "none";
      var feedback_answer = document.getElementById("feedback_answer");
      feedback_answer.style.display = "block";
      // now check whether human correct or not
      // personAnswers[count] is the answer that the person gave, it is either yes or no or ai

      // Retrieve elements with IDs "correct_overall_or_no" and "was_ai_correct"
      var overallHeader = document.getElementById("correct_overall_or_no");
      var aiHeader = document.getElementById("was_ai_correct");
      var userAnswer;
      // If user answered "ai", retrieve AI answer from variable
      if (selection === "ai") {
        userAnswer = ai_answers_raw[count];
      } 

      // convert labels_raw[count] to string
      var label_in_text = labels_raw[count].toString();
      
      // Compare user answer to label_raw
      if (userAnswer === labels_raw[count]) {
        overallHeader.textContent =
          "Correct: You predicted correctly!" +
          ", the correct answer was " +
          label_in_text +
          ".";
        // change image image_feedback to correct.png
        document.getElementById("image_feedback").src = "../correct.png";
      } else {
        overallHeader.textContent =
          "Wrong: You predicted incorrectly." +
          ", the correct answer was " +
          label_in_text +
          ".";
        document.getElementById("image_feedback").src = "../incorrect.png";
      }

      if (ai_answers_raw[count] === labels_raw[count]) {
        aiHeader.textContent = "AI was Correct: The AI predicted correctly.";
      } else {
        aiHeader.textContent =
          "AI was Incorrect: The AI predicted incorrectly.";
      }
    }
  }
};

nextButton.onclick = function () {
  if (count < images.length - 1) {
    count += 1;
    console.log(selection);
    getStorageImage(images[count], ai_images[count]);
    loadAiAnswer(ai_answers[count], ai_answers_raw[count]);
    updateProgressBar();
    var prediction_interface = document.getElementById("prediction_interface");
    prediction_interface.style.display = "block";
    var feedback_answer = document.getElementById("feedback_answer");
    feedback_answer.style.display = "none";
  } else {
    // go to end task page
    disableBeforeUnload();

      window.location.href = "./testing_task_ai.html";
    
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
      teaching_answers_baseline: personAnswers,
      teaching_times_baseline: test_pressed_times,
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
  var elem_feedback = document.getElementById("ai_answer_feedback");
  elem.innerHTML = ai_answer;
  elem_feedback.innerHTML = ai_answer;
  // if ai_answer_raw == 1, then make background rgb(135, 208, 135), else rgb(247, 84, 84)

}

function loadTaskData() {
  db.collection("tasks_mmlu")
    .doc(task_id)
    .get()
    .then(function (query_snapshot) {
      rand_task = query_snapshot;
      images = query_snapshot.data().teaching_baseline_images;
      ai_images = query_snapshot.data().teaching_baseline_images_box;
      ai_answers = query_snapshot.data().teaching_baseline_ai_answers;
      ai_answers_raw = query_snapshot.data().teaching_baseline_ai_answers_raw;
      labels_raw = query_snapshot.data().teaching_baseline_labels;
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
enableBeforeUnload();
