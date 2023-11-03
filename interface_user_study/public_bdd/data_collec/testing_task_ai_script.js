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
let time = 30;
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
var yesdButton = document.getElementById("yesButton");
var noButton = document.getElementById("noButton");
var submitButton = document.getElementById("submitButton");
var hideButton = document.getElementById("hideButton");
var AIButton = document.getElementById("AIButton");
var img = document.getElementById("myimg");
var imgAI = document.getElementById("myimgAI");

let selection;
yesButton.onclick = function () {
  selection = "yes";
};
noButton.onclick = function () {
  selection = "no";
};
AIButton.onclick = function () {
  selection = "ai";
};
// if none of the buttons are clicked, selection is undefined


function resetShowHideButton() {
  hideButton.innerHTML = "Hide AI answer";
  img.style = "display:none;";
  imgAI.style = "display:block;";
  ai_hidden = 0;
}


hideButton.onclick = function () {
  if (ai_hidden == 0) {
    ai_hidden = 1;
    imgAI.style = "display:none;";
    img.style = "display:block;";
    hideButton.innerHTML = "Show AI answer";
  } else {
    ai_hidden = 0;
    img.style = "display:none;";
    imgAI.style = "display:block;";
    hideButton.innerHTML = "Hide AI answer";
  }
};

submitButton.onclick = function () {
  // check if one of the buttons is clicked
  if ( yesButton.disabled == true && noButton.disabled == true && AIButton.disabled == true) {
    alert("Please choose an answer");
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
      resetShowHideButton();
      updateProgressBar();
    } else {
      // go to end task page
      disableBeforeUnload();
      if (treatment == 1) {
      window.location.href = "../end_task.html";
      }
      else {
        window.location.href = "./testing_task_noai.html";
      }
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
  db.collection("responses")
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
var imgElement = document.getElementById('myimg');
imgElement.src = "../Loading.gif";
var imgElement = document.getElementById('myimgAI');
imgElement.src = "../Loading.gif";
  var imageRef = storage.ref("/bdd_images/" + image_name);
  // Get the download URL of the image
  imageRef
    .getDownloadURL()
    .then(function (url) {
      // Set the URL as the source of the img element to display the image
      var imgElement = document.getElementById("myimg");
      imgElement.src = url;
    })
    .catch(function (error) {
      // Handle any errors
      console.error(error);
    });
  var imageRef = storage.ref("/bdd_images/" + ai_image_name);
  // Get the download URL of the image
  imageRef
    .getDownloadURL()
    .then(function (url) {
      // Set the URL as the source of the img element to display the image
      var imgElement = document.getElementById("myimgAI");
      imgElement.src = url;
    })
    .catch(function (error) {
      // Handle any errors
      console.error(error);
    });
}

function loadAiAnswer(ai_answer, ai_answer_raw) {
  var elem = document.getElementById("ai_answer");
  elem.innerHTML = ai_answer;
  // if ai_answer_raw == 1, then make background rgb(135, 208, 135), else rgb(247, 84, 84)
  if (ai_answer_raw == 1) {
    elem.style.backgroundColor = "#87d087";
  }
  else {
    elem.style.backgroundColor = "#f75454";
  }
  
}

function loadTaskData() {
  db.collection("tasks")
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
enableBeforeUnload();
resetShowHideButton()