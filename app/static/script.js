const input = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const predictBtn = document.getElementById("predictBtn");
const result = document.getElementById("result");

input.addEventListener("change", function () {

    const file = input.files[0];

    if (!file) return;

    preview.src = URL.createObjectURL(file);
    preview.style.display = "inline-block";

    document.getElementById("fileName").innerText =
        file.name;

    predictBtn.disabled = false;

    result.style.display = "none";
});


async function predictDisease() {

    const file = input.files[0];

    if (!file) return;

    predictBtn.disabled = true;

    document.getElementById("loading").innerText =
        "🔄 AI is analyzing the leaf...";

    const formData = new FormData();

    formData.append("file", file);

    try {

        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error("Prediction failed");
        }

        const data = await response.json();

        const predictions = data.predictions;

        // Top prediction
        const top = predictions[0];

        const percentage = top.confidence * 100;

        document.getElementById("disease").innerText =
            top.class;

        document.getElementById("confidence").innerText =
            percentage.toFixed(2) + "%";

        document.getElementById("confidenceBar").style.width =
            percentage + "%";


        // Top 3
        const predictionList =
            document.getElementById("predictionList");

        predictionList.innerHTML = "";

        predictions.forEach((prediction, index) => {

            const percent =
                prediction.confidence * 100;

            const div =
                document.createElement("div");

            div.className = "prediction";

            div.innerHTML = `
                <span class="prediction-name">
                    ${index + 1}. ${prediction.class}
                </span>

                <span class="prediction-confidence">
                    ${percent.toFixed(2)}%
                </span>
            `;

            predictionList.appendChild(div);

        });


        // Warning
        const warning =
            document.getElementById("warning");

        if (percentage < 50) {

            warning.className = "warning";

            warning.innerText =
                "⚠️ Low confidence prediction. Try uploading a clearer leaf image.";

        } else {

            warning.className = "";

            warning.innerText = "";
        }


        result.style.display = "block";

        result.scrollIntoView({
            behavior: "smooth"
        });

    }

    catch (error) {

        document.getElementById("loading").innerText =
            "❌ Prediction failed. Please try again.";

        console.error(error);

    }

    predictBtn.disabled = false;

    setTimeout(() => {
        document.getElementById("loading").innerText = "";
    }, 2000);
}