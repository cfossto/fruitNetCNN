let uploadProgress = []
let progressBar = document.getElementById('progress-bar')



function previewFile(file) {
  let reader = new FileReader()
  reader.readAsDataURL(file)
  reader.onloadend = function() {
    let img = document.createElement('img')
    img.src = reader.result
    document.getElementById("gallery").insertAdjacentHTML("beforeend", "<div class='popup' onclick='answer_cnn()'>         <span class='popuptext' id='myPopup'></span></div>")
    document.getElementById('gallery').appendChild(img)
  }
}

function answer_cnn() {
    var popup = document.getElementById("myPopup");
    popup.classList.toggle("show");
}


let dropArea = document.getElementById('drop-area')

  ;['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, preventDefaults, false)
  })
  
function preventDefaults (e) {
    e.preventDefault()
    e.stopPropagation()
  }

  ;['dragenter', 'dragover'].forEach(eventName => {
    dropArea.addEventListener(eventName, highlight, false)
  })
  
  ;['dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, unhighlight, false)
  })
  
function highlight(e) {
    dropArea.classList.add('highlight')
  }
  
function unhighlight(e) {
    dropArea.classList.remove('highlight')
  }

dropArea.addEventListener('drop', handleDrop, false)

function handleDrop(e) {
  let dt = e.dataTransfer
  let files = dt.files

  handleFiles(files)
}

function handleFiles(files){
    files = [...files]
    initializeProgress(files.lenght)
    files.forEach(previewFile)
    files.forEach(uploadFile)
  }



function uploadFile(file, i) {
    let url = '/imageSend'
    var xhr = new XMLHttpRequest()
    let formData = new FormData()
    xhr.open('POST', url, true)

    
    xhr.upload.addEventListener("progress", function(e) {
      updateProgress(i, (e.loaded * 100.0 / e.total) || 100)
    })
  
    xhr.addEventListener('readystatechange', function(e) {
      if (xhr.readyState == 4 && xhr.status == 200) {
        let response = JSON.parse(this.responseText)
        console.log(response)
        document.querySelector("#myPopup").insertAdjacentHTML("beforeend",`<p>This is a ${response.prediction}</p>`)
        answer_cnn()
        document.querySelector("#gallery").insertAdjacentHTML("afterend",`<p>We are  ${response.confidence}% certain`)
        // Done. Inform the user
      }
      else if (xhr.readyState == 4 && xhr.status != 200) {
        // Error. Inform the user
      }
    })
  
    formData.append('file', file)
    xhr.send(formData)

  
    .then(updateProgress)
    .catch(() => { /* Error. Inform the user */ })
  }

function initializeProgress(numFiles) {
    progressBar.value = 0
    uploadProgress = []

    for(let i = numFiles; i > 0; i--) {
        uploadProgress.push(0)
    }
  }

function updateProgress(fileNumber, percent) {
    uploadProgress[fileNumber] = percent
    let total = uploadProgress.reduce((tot, curr) => tot + curr, 0) / uploadProgress.length
    progressBar.value = total
}
  
function progressDone() {
    filesDone++
    progressBar.value = filesDone / filesToDo * 100
  }


