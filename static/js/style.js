"use strict";

// Update  Color



var list = document.getElementsByTagName("td");

for (const elem of list) {
    // console.log(elem.innerHTML);
    // console.log(elem.style);
    if (elem.innerHTML === "Fraud"){
      elem.style.color = "black";
      elem.style.backgroundColor="red";
    }
    else if (elem.innerHTML === "Not fraud"){
      elem.style.color= "white";
      elem.style.backgroundColor = "green";
    }}



    $("td").click(function(){

        var value=$(this).parent().siblings(":first").text()
       
        alert(value)  ;

        $(this).parent().remove()
       
       });
  