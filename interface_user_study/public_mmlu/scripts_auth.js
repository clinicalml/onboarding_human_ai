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
var authen_token;

var is_valid = false;

/* firebase
  .auth()
  .signOut()
  .then(() => {
    console.log("signed out");
  })
  .catch((error) => {
    console.log(error);
  }); */

document.getElementById("submit_token").onclick = function () {
  var token_user = document.getElementById("authen_token").value;
  console.log(token_user);
  var email = "prolific_user@gmail.com";

  firebase
    .auth()
    .signInWithEmailAndPassword(email, token_user)
    .then((userCredential) => {
      // Signed in teachingtodeferexperiment
      // swich window
      location.href = "./worker_info.html";
      
      // ...
    })
    .catch((error) => {
      var errorCode = error.code;
      var errorMessage = error.message;
      var error_answer = document.getElementById("message_highlighted");
      error_answer.innerHTML = "Invalid Token, please try again.";
    });
};

