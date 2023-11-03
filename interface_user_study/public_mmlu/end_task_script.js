
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

var true_ans;
var started_task = false;


function showlocalstorage() {
  console.log(response_id);
  console.log(task_id);
}

function loadlocalstorage() {
  var myData = localStorage['objectToPass'];
  myData = JSON.parse(myData);
  response_id = myData[0];
  task_id = myData[1];
}



document.getElementById("proceed_exp").onclick = function() {

  var quest1 = document.getElementById("quest1").value;
  var quest2 = document.getElementById("quest2").value;

  if (firebase.auth().currentUser){
    console.log("logged in");
  }
  if (quest1 != "" && quest2 != "") {
    db.collection("responses_mmlu").doc(response_id).update({
        outake_quest1: quest1,
        outake_quest2: quest2,
        completed_task: 1
      })
      .then(() => {
        console.log("Document successfully written!");
        document.getElementById("body_end").style.display = "none";
        document.getElementById("header_end").innerHTML = "Please copy the code below into the Prolific stuy on the Prolific site and then you can close this window as the task is over.";
        document.getElementById("header_end").innerHTML += "<br> If you do not copy the code, your study will not be approved.";
        var worker_id = response_id.split("-");
        document.getElementById("worker_id_send").innerHTML = "C1I4M3L1";

        firebase.auth().signOut().then(() => {
          console.log("signed out");
        }).catch((error) => {
          console.log(error);
        });


      })
      .catch((error) => {
        console.error("Error writing document: ", error);
      });
  } else {
    document.getElementById("message_highlighted").innerHTML = "Please answer all questions";
  }
};


loadlocalstorage();
