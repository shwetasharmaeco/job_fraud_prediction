"use strict";

// Update  Color

var list = document.getElementsByTagName("td");

for (const elem of list) {
    
    if (elem.innerHTML === "Fraud"){
      elem.style.color = "black";
      elem.style.backgroundColor="red";
    }
    else if (elem.innerHTML === "Not fraud"){
      elem.style.color= "white";
      elem.style.backgroundColor = "green";
    }}



// confirm deleting a job posting


    $("td").click(function(){
    var value=$(this).parent().text()
    if(confirm("Are you sure you want to delete this?")){
        $("td").attr("href", "query.php?ACTION=delete&ID='1'");
        $(this).parent().remove();
    }
    else{
        return false;
    }
});


// storing all fraud jobs rows

function getReport() {
    var frauds =[];
    $("td:nth-child(1)").each(function () {
        if ($(this).text() === "Fraud") {
            frauds.push($(this).parent().text());
            console.log(frauds);
        }
    });
    localStorage.setItem('todoList', JSON.stringify(frauds));
};



var time = new Date();

localStorage.setItem('time', time);



