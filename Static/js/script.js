/**
 * PriceSense Dashboard - Client Interactions Engine
 */
document.addEventListener('DOMContentLoaded', () => {
    initializeCardAnimations();
    initializeMarketMetrics();
    enhanceBootstrapAlerts();
});

/**
 * Applies a smooth parallax intersection fade-in effect when user scrolls to dashboard elements
 */
function initializeCardAnimations() {
    const targets = document.querySelectorAll('.dashboard-card, .stats-card, .market-card, .graph-preview, .overview-card, .menu-card, .latest-card');
    
    const observerOptions = {
        root: null,
        threshold: 0.05,
        rootMargin: '0px 0px -20px 0px'
    };

    const observer = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    targets.forEach(target => {
        // Set styling base values before executing animation sequence
        target.style.opacity = '0';
        target.style.transform = 'translateY(25px)';
        target.style.transition = 'opacity 0.6s cubic-bezier(0.25, 1, 0.5, 1), transform 0.6s cubic-bezier(0.25, 1, 0.5, 1)';
        observer.observe(target);
    });
}

/**
 * Animates text counters or scales market numbers for premium presentation layers
 */
function initializeMarketMetrics() {
    const brentCrudeBox = document.querySelector('.market-box h4');
    if (brentCrudeBox && brentCrudeBox.textContent.includes('$')) {
        // Highlights the price point subtly on active dashboard rendering
        brentCrudeBox.style.transition = 'color 0.3s ease';
        brentCrudeBox.style.color = '#10b981'; // Shifts color dynamically to an emerald accent tone
    }
}

/**
 * Configures flash messages to gracefully melt away instead of abruptly snapping closed
 */
function enhanceBootstrapAlerts() {
    const alerts = document.querySelectorAll('.alert-dismissible');
    alerts.forEach(alert => {
        setTimeout(() => {
            alert.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
            alert.style.opacity = '0';
            alert.style.transform = 'translateY(-10px)';
            setTimeout(() => {
                alert.remove();
            }, 500);
        }, 5000); // Fades warning messages away completely after 5 consecutive seconds
    });
}

