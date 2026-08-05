/*==================================================
            PRICESENSE - JAVASCRIPT
    Dynamic Fuel Price Forecast Platform
==================================================*/


/*==================================================
            PAGE LOADED MESSAGE
==================================================*/

document.addEventListener("DOMContentLoaded", function () {

    console.log("PriceSense Website Loaded Successfully.");

});


/*==================================================
            AUTO HIDE FLASH MESSAGE
==================================================*/

const alerts = document.querySelectorAll(".alert");

alerts.forEach(function(alert){

    setTimeout(function(){

        alert.style.opacity = "0";

        setTimeout(function(){

            alert.style.display = "none";

        },500);

    },3000);

});


/*==================================================
            INPUT FOCUS EFFECT
==================================================*/

const inputs = document.querySelectorAll("input");

inputs.forEach(function(input){

    input.addEventListener("focus", function(){

        this.parentElement.classList.add("active");

    });

    input.addEventListener("blur", function(){

        this.parentElement.classList.remove("active");

    });

});


/*==================================================
            SIDEBAR ACTIVE LINK
==================================================*/

const menuLinks = document.querySelectorAll(".sidebar ul li a");

menuLinks.forEach(function(link){

    link.addEventListener("click", function(){

        menuLinks.forEach(function(item){

            item.parentElement.classList.remove("active");

        });

        this.parentElement.classList.add("active");

    });

});

/*==================================================
            SCROLL TO TOP BUTTON
==================================================*/

const scrollButton = document.getElementById("scrollTopBtn");

window.addEventListener("scroll", function () {

    if (scrollButton) {

        if (window.scrollY > 250) {

            scrollButton.style.display = "block";

        }

        else {

            scrollButton.style.display = "none";

        }

    }

});


if (scrollButton) {

    scrollButton.addEventListener("click", function () {

        window.scrollTo({

            top: 0,

            behavior: "smooth"

        });

    });

}


/*==================================================
            CARD HOVER EFFECT
==================================================*/

const cards = document.querySelectorAll(

    ".dashboard-card, .menu-card, .graph-card, .card, .result-card"

);

cards.forEach(function(card){

    card.addEventListener("mouseenter", function(){

        this.style.transform = "translateY(-8px)";

    });

    card.addEventListener("mouseleave", function(){

        this.style.transform = "translateY(0px)";

    });

});


/*==================================================
            IMAGE FADE EFFECT
==================================================*/

const images = document.querySelectorAll("img");

images.forEach(function(image){

    image.style.opacity = "0";

    image.style.transition = "opacity 0.8s ease";

    image.onload = function(){

        image.style.opacity = "1";

    };

});


/*==================================================
            DASHBOARD NUMBER ANIMATION
==================================================*/

const numbers = document.querySelectorAll(".dashboard-card h3");

numbers.forEach(function(number){

    const target = parseFloat(number.innerText);

    if(!isNaN(target)){

        let count = 0;

        const speed = target / 40;

        const updateCounter = function(){

            if(count < target){

                count += speed;

                number.innerText = Math.round(count);

                requestAnimationFrame(updateCounter);

            }

            else{

                number.innerText = target;

            }

        };

        updateCounter();

    }

});

/*==================================================
        FORM VALIDATION
==================================================*/

const predictionForm = document.querySelector("form");

if (predictionForm) {

    predictionForm.addEventListener("submit", function (event) {

        const inputs = predictionForm.querySelectorAll("input[required]");

        let valid = true;

        inputs.forEach(function (input) {

            if (input.value.trim() === "") {

                valid = false;

                input.style.border = "2px solid red";

            }

            else {

                input.style.border = "2px solid #2563EB";

            }

        });

        if (!valid) {

            event.preventDefault();

            alert("Please fill all required fields.");

        }

    });

}


/*==================================================
        RESET FORM CONFIRMATION
==================================================*/

const resetButton = document.querySelector(".btn-secondary");

if (resetButton) {

    resetButton.addEventListener("click", function () {

        setTimeout(function () {

            alert("Form has been cleared successfully.");

        }, 100);

    });

}


/*==================================================
        PREDICT BUTTON LOADING
==================================================*/

const submitButton = document.querySelector(".btn-primary");

if (predictionForm && submitButton) {

    predictionForm.addEventListener("submit", function () {

        submitButton.disabled = true;

        submitButton.innerHTML =

            '<i class="fa-solid fa-spinner fa-spin"></i> Predicting...';

    });

}


/*==================================================
        SMOOTH SCROLL
==================================================*/

document.querySelectorAll("a[href^='#']").forEach(function(anchor){

    anchor.addEventListener("click", function(e){

        e.preventDefault();

        const target = document.querySelector(this.getAttribute("href"));

        if(target){

            target.scrollIntoView({

                behavior:"smooth"

            });

        }

    });

});


/*==================================================
        CONSOLE MESSAGE
==================================================*/

console.log("PriceSense JavaScript Loaded Successfully.");

