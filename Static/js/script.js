/*==================================================
            PRICESENSE JAVASCRIPT
        Dynamic Fuel Price Forecast Platform
==================================================*/

document.addEventListener("DOMContentLoaded", function () {

    console.log("PriceSense Loaded Successfully");
    // Auto Close Alerts
    const alerts = document.querySelectorAll(".alert");

    alerts.forEach(function(alert){

        setTimeout(function(){

            alert.style.transition="0.5s";
            alert.style.opacity="0";

            setTimeout(function(){

                alert.remove();

            },500);

        },3000);

    });
    /*==================================================
            ANIMATED COUNTERS
==================================================*/

    const counters = document.querySelectorAll(".stats-card h2, .dashboard-card h3");

    counters.forEach(counter => {

        const target = counter.innerText;

        if(!isNaN(parseFloat(target))){

            let count = 0;
            const end = parseFloat(target);

            const speed = end / 80;

            const updateCounter = () =>{

                count += speed;

                if(count < end){

                    counter.innerText = count.toFixed(0);

                    requestAnimationFrame(updateCounter);

                }

                else{

                    if(target.includes("%")){

                        counter.innerText = end.toFixed(2) + "%";

                    }

                    else{

                        counter.innerText = target;

                    }

                }

            };

            updateCounter();

        }

    });
    /*==================================================
            SCROLL REVEAL ANIMATION
==================================================*/

    const revealItems = document.querySelectorAll(

    ".dashboard-card,\
    .stats-card,\
    .market-box,\
    .menu-card,\
    .latest-card,\
    .technology-card,\
    .graph-card,\
    .overview-card"

    );

    const reveal = () =>{

        revealItems.forEach(item=>{

            const top = item.getBoundingClientRect().top;

            const windowHeight = window.innerHeight;

            if(top < windowHeight - 80){

                item.classList.add("show-card");

            }

        });

    };

    window.addEventListener("scroll", reveal);

    reveal();

    /*==================================================
                BUTTON RIPPLE EFFECT
    ==================================================*/

    const buttons = document.querySelectorAll(

    ".btn-primary,\
    .login-btn,\
    .menu-card,\
    .btn-danger"

    );

    buttons.forEach(button=>{

        button.addEventListener("click",function(e){

            const circle=document.createElement("span");

            const diameter=Math.max(

                this.clientWidth,
                this.clientHeight

            );

            circle.style.width=diameter+"px";
            circle.style.height=diameter+"px";

            circle.style.left=e.offsetX-diameter/2+"px";
            circle.style.top=e.offsetY-diameter/2+"px";

            circle.classList.add("ripple");

            const ripple=this.getElementsByClassName("ripple")[0];

            if(ripple){

                ripple.remove();

            }

            this.appendChild(circle);

        });

    });
    /*==================================================
                    PAGE FADE EFFECT
    ==================================================*/

    document.body.style.opacity="0";

    setTimeout(function(){

        document.body.style.transition="opacity .8s";

        document.body.style.opacity="1";

    },100);
    
});

