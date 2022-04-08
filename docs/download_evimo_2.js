var coll = document.getElementsByClassName("collapsible");
var i;

for (i = 0; i < coll.length; i++) {
  coll[i].addEventListener("click", function() {
    this.classList.toggle("active");
    var content = this.nextElementSibling;
    if (content.style.display === "block") {
      content.style.display = "none";
      var video = content.getElementsByTagName("video")
      if (video.length == 1) {
        video[0].pause()
      }
    } else {
      content.style.display = "block";
      var video = content.getElementsByTagName("video")
      if (video.length == 1) {
        video[0].play()
      }
    }
  });
}
