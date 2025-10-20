(function() {
    // Helper function to extract hostname from URL
    function getHost(url) {
        var a = document.createElement('a');
        a.href = url;
        return a.hostname;
    }

    // Function to check if access is allowed based on the call stack
    function isAllowed() {
        // Extract all URLs from the stack trace
        var regex = /(https?:\/\/.+?):\d+:\d+/g;
        var stack = new Error().stack;
        var urls = stack.match(regex);
        if (urls && urls.length > 0) {
            // Use last entry and extract its hostname from URL
            var topCaller = getHost(urls[urls.length - 1]);
            return topCaller === window.location.hostname;
        }
        return true;
    }

    // Store the original cookie property descriptor
    var originalCookieDescriptor = Object.getOwnPropertyDescriptor(Document.prototype, 'cookie');

    // Define a new property with custom getter and setter
    Object.defineProperty(document, 'cookie', {
        // Custom getter
        get: function() {
            if (isAllowed()) {
                return originalCookieDescriptor.get.call(document);
            } else {
                console.log('Access denied: cookie read access from third-party script');
                return ''; // Restrict access
            }
        },
        // Custom setter
        set: function(value) {
            if (isAllowed()) {
                originalCookieDescriptor.set.call(document, value);
            } else {
                console.log('Access denied: cookie write access from third-party script');
                // Optionally, restrict setting the cookie
            }
        }
    });
})();