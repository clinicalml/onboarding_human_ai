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
var exp_condition;
document.getElementById("tasks").style.display = "none";

document.getElementById("wrong-message").style.display = "none";

var timedButton = document.getElementById("test_timer_button");
let time = 10;
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
}, 1000);

timedButton.onclick = function () {
  document.getElementById("instructions").style.display = "none";
  document.getElementById("tasks").style.display = "block"; // show first task
  console.log(time);
  document.getElementById("test_timer_button").style.display = "none";
  return false; // stops page from reloading
};

let count = 0;
var errors = 0;
var images;
var answers;
var exp_thing = 1;

let personAnswers = [];

var yesdButton = document.getElementById("yesButton");
var noButton = document.getElementById("noButton");
var submitButton = document.getElementById("submitButton");

let selection;
yesButton.onclick = function () {
  selection = "yes";
};
noButton.onclick = function () {
  selection = "no";
};

document.getElementById("submitButton").onclick = function () {
  console.log("push", selection, count, answers[count]);
  if (selection == undefined) {
    alert("Please choose an answer");
  } else {
    personAnswers.push(selection);
    if (selection != answers[count] && count < images.length) {
      document.getElementById("wrong-message").style.display = "block";
      errors += 1;
      if (errors == 3) {
        disableBeforeUnload();
        // end the experiment for user
        location.href = "../invalid.html";
        db.collection("responses").doc(response_id).update({
          completed_task: -1,
          errors_task: -1,
        });
      }
      // while incorrect, show message
    } else if (count == images.length - 1) {
      // sample a binary number
      var treatment = Math.floor(Math.random() * 2);
      db.collection("responses")
        .doc(response_id)
        .update({
          random_order: treatment,
          exp_thing: exp_thing,
        })
        .then(() => {
          var myData = [response_id, task_id, exp_condition, treatment];
          disableBeforeUnload();
          localStorage.setItem("objectToPass", JSON.stringify(myData));
          location.href = "./testing_task_noai.html";
          if (exp_condition == 1) {
            if (treatment == 1) {
              location.href = "./data_collec/testing_task_noai.html";
            } else {
              location.href = "./data_collec/testing_task_ai.html";
            }
          }
          if (exp_condition == 2) {
            if (exp_thing == 1) {
              location.href = "./teaching/teaching_task.html";
            } else if (exp_thing == 2) {
              location.href = "./teach_example/teaching_task.html";
            } else if (exp_thing == 3) {
              location.href = "./test_rec/testing_task_ai_rec.html";
            } else if (exp_thing == 4) {
              if (treatment == 1) {
                location.href = "./data_collec/testing_task_noai.html";
              } else {
                location.href = "./data_collec/testing_task_ai.html";
              }
            }
          }
        })
        .catch((error) => {
          console.error("Error writing document: ", error);
        });
    } else {
      count += 1;
      getStorageImage(images[count]);
      updateProgressBar();
      document.getElementById("wrong-message").style.display = "none";
    }
    selection = undefined;
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
  exp_condition = myData[2];
  //showlocalstorage();
}

function getStorageImage(image_name) {
  var imgElement = document.getElementById("myimg");
  imgElement.src = "./Loading.gif";
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
}

function updateProgressBar() {
  var elem = document.getElementById("prog_bar");
  var width = (count / images.length) * 100;
  elem.style.width = width + "%";
  elem.innerHTML = count + "/" + images.length;
}

function loadTaskData() {
  db.collection("tasks")
    .doc(task_id)
    .get()
    .then(function (query_snapshot) {
      rand_task = query_snapshot;
      images = query_snapshot.data().welcome_images;
      answers = query_snapshot.data().welcome_answers;
      exp_condition = query_snapshot.data().exp_condition;
      // get image from firebase storage
      getStorageImage(images[0]);
    })
    .catch(function (error) {
      console.log("Error getting documents: ", error);
    });
}
function enableBeforeUnload() {
  window.onbeforeunload = function (e) {
    return "Discard changes?";
  };
}
function disableBeforeUnload() {
  window.onbeforeunload = null;
}

enableBeforeUnload();
loadlocalstorage();
loadTaskData();
