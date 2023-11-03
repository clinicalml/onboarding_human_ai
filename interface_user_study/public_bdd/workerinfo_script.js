// Import the functions you need from the SDKs you need
// Your web app's Firebase configuration
// Your web app's Firebase configuration
var firebaseConfig = {

};

// Initialize Firebase
firebase.initializeApp(firebaseConfig);
firebase.analytics();


// source: https://stackoverflow.com/questions/69386541/how-to-disable-submit-button-until-all-mandatory-fields-are-filled-using-html-an

var db = firebase.firestore();
var task_id_rand;
var worker_id_rand = Math.floor(Math.random() * 10000000); // to pass to other pages
var rand_task;
var response_id;
var exp_condition = 2;

// we retreive task from database here add conditions
db.collection("tasks")
  .where("exp_condition", "==", exp_condition)
  .get()
  .then(function(query_snapshot) {
    rand_task = query_snapshot.docs[Math.floor(Math.random() * query_snapshot.docs.length)];
    task_id_rand = rand_task.id;
    exp_condition = rand_task.data().exp_condition;
  })
  .catch(function(error) {
    console.log("Error getting documents: ", error);
  });

let inputs = document.querySelectorAll("input");
let selections = document.querySelectorAll("select");
let buttonSend = document.getElementById("button-send");
// submitButton
let submitButton = document.getElementById("submitButton");
let validator = {
  workerID: false,
  age: false,
  gender: false,
  education: false,
  ai_ability: false,
  wiki_often: false,
};

let workerID_input = document.getElementById("workerID");

function isNotEmpty(input) {
  // Check if the input value is empty or not
  if (input.value.length == 0) {
    // If empty, return false and show an error message
    alert("Please fill in this field");
    return false;
  }
  // If not empty, return true
  return true;
}

function isNotEmpty(selection) {
  // Check if the input value is empty or not
  if (selection.value.length == 0) {
    // If empty, return false and show an error message
    alert("Please fill in this field");
    return false;
  }
  // If not empty, return true
  return true;
}

function submit(event) {
  event.preventDefault();

  var name_worker = document.getElementById("workerID").value;
  var age_worker = document.getElementById("age").options[document.getElementById("age").selectedIndex].text;
  var gender_worker = document.getElementById("gender").options[document.getElementById("gender").selectedIndex].text;
  var education_worker = document.getElementById("education").options[document.getElementById("education").selectedIndex].text;
  var ai_ability_worker = document.getElementById("ai_ability").options[document.getElementById("ai_ability").selectedIndex].text;
  var eyesight_worker = document.getElementById("eyesight").options[document.getElementById("eyesight").selectedIndex].text;
  var driving_worker = document.getElementById("driving").options[document.getElementById("driving").selectedIndex].text;
 
  var radios_checked = true;



  if (firebase.auth().currentUser && name_worker != "" && radios_checked == true) {
    // create new doc
    var worker_in_responses = true;
    response_id = task_id_rand.concat("-").concat(worker_id_rand.toString());
    // get time now in string format month day hour and minutes secs
    var date = new Date();
    var date_string = date.getMonth().toString().concat("-").concat(date.getDate().toString()).concat("-").concat(date.getHours().toString()).concat("-").concat(date.getMinutes().toString()).concat("-").concat(date.getSeconds().toString());
    console.log(date_string);
    db.collection("responses").where("name", "==", name_worker)
      .get()
      .then((querySnapshot) => {
        if (querySnapshot.docs.length == 0) { // only if worker has not filled it out yet
          worker_in_responses = false;
          db.collection("responses").doc(response_id).set({
              worker_id: worker_id_rand,
              task_id: task_id_rand,
              name: name_worker,
              date_performed: date_string,
              age_worker: age_worker,
              gender_worker: gender_worker,
              ai_worker: ai_ability_worker,
              education_worker: education_worker,
              eyesight_worker: eyesight_worker,
              driving_worker: driving_worker,
              completed_task: 0,
              exp_condition: exp_condition
            })
            .then(() => {
              console.log("Document successfully written!");
              var myData = [response_id, task_id_rand, exp_condition];
              localStorage.setItem('objectToPass', JSON.stringify(myData));
              location.href = "./welcome_task.html";
            })
            .catch((error) => {
              console.error("Error writing document: ", error);
            });
        } else {
          worker_in_responses = true;
          var error_answer = document.getElementById("message_highlighted");
          error_answer.innerHTML = "Already completed task, cannot perform task again.";
        }

      })
      .catch((error) => {
        worker_in_responses = true;
      });
   
  } else {
    console.log("error in filling out form");
    var error_answer = document.getElementById("message_highlighted");
    error_answer.innerHTML = "Not signed in or information missing";
  }  
  return false;
}

var form = document.getElementById("form");
form.addEventListener('submit', submit);



function disableBeforeUnload() {
  window.onbeforeunload = null;
}

function enableBeforeUnload() {
  window.onbeforeunload = function (e) {
    return "Discard changes?";
  };
}

//enableBeforeUnload();
