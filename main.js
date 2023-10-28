const resultContainerElement = document.querySelector('#result-container');
const textElement = document.querySelector('#text');
document.querySelector('#submit-btn').addEventListener('click', (event) => {
    event.preventDefault();
    const textForCheck = textElement.value;

    fetch('http://localhost:3200/sarcasm_percentage',{
        method: 'POST',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            text: textForCheck
        })
    }).then((response) => response.text()).then((data) => {
        console.log(data);
        resultContainerElement.innerText = data;

    }).catch((err) => {
        console.error(err);
    })});