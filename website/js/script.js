// Function to generate SHA-256 hash of text using crypto-js
function sha256(text) {
    const hash = CryptoJS.SHA256(text);
    return hash.toString(CryptoJS.enc.Hex);
}

// Enhanced fingerprinting function with canvas fingerprinting
async function getFingerprint() {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 200;
    canvas.height = 60;
    ctx.textBaseline = "top";
    ctx.font = "14px 'Arial'";
    ctx.textBaseline = "alphabetic";
    ctx.fillStyle = "#f60";
    ctx.fillRect(125, 1, 62, 20);
    ctx.fillStyle = "#069";
    ctx.fillText("Hello, world!", 2, 15);
    ctx.fillStyle = "rgba(102, 204, 0, 0.7)";
    ctx.fillText("Hello, world!", 4, 17);

    const dataUrl = canvas.toDataURL();
    const canvasHash = sha256(dataUrl);


    const fingerprint = {
        userAgent: navigator.userAgent,
        screenSize: `${screen.width}x${screen.height}`,
        language: navigator.language,
        canvasHash: canvasHash, // Use the hash as a shorter representation
    };
    return fingerprint;
}

// Function to display results
async function displayResults(data) {
    const resultsElement = document.getElementById('results');
    resultsElement.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
}

// Main function adjusted for asynchronous getFingerprint
async function main() {
    const fingerprint = await getFingerprint();
    console.log(fingerprint);
    await displayResults(fingerprint);
}

// Execute the main function
main();
