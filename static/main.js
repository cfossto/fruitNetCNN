let uploadProgress = []
let progressBar = document.getElementById('progress-bar')


function sendFile(){
    console.log("works!")
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

function handleFiles(files) {
    files = [...files]
    initializeProgress(files.lenght)
    files.forEach(uploadFile)
    files.forEach(previewFile)
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
        // Done. Inform the user
      }
      else if (xhr.readyState == 4 && xhr.status != 200) {
        // Error. Inform the user
      }
    })
  
    formData.append('file', file)
    xhr.send(formData)
  
    fetch(url, {
      method: 'POST',
      body: formData
    })
    .then(updateProgress)
    .catch(() => { /* Error. Inform the user */ })
  }

function previewFile(file) {
    let reader = new FileReader()
    reader.readAsDataURL(file)
    reader.onloadend = function() {
      let img = document.createElement('img')
      img.src = reader.result
      document.getElementById('gallery').appendChild(img)
    }
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

