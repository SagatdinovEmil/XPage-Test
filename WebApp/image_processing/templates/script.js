const dropzone = document.getElementById('image-dropzone');
const imageInput = document.getElementById('image');
const imagePreview = document.getElementById('image-preview');
const previewImage = imagePreview.querySelector('img');

imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.style.display = 'block';
            previewImage.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
});

dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');

    const file = e.dataTransfer.files[0];
    if (file) {
        imageInput.files = e.dataTransfer.files;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.style.display = 'block';
            previewImage.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
});
