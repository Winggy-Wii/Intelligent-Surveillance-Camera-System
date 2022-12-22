var first_img = document.getElementById("first-cam-imgs")
var second_img = document.getElementById("second-cam-imgs")
var baseHost = document.location.origin;

function FindPosition(oElement)
{
  if(typeof( oElement.offsetParent) != "undefined"){
    for(var posX = 0, posY = 0; oElement; oElement = oElement.offsetParent)
    {
      posX += oElement.offsetLeft;
      posY += oElement.offsetTop;
    }
    return [posX, posY];
    
  }
  else{
    posX = oElement.x;
    posY = oElement.y
    return [posX, posY];
    
  }
}

function GetCoordinates(event)
{
  let PosX = 0;
  let PosY = 0;
  let ImgWidth = 0;
  let ImgHeight = 0;
  let ImgPos;
  ImgPos = FindPosition(first_img)
  console.log(ImgPos);
    PosX = event.clientX;
    PosY = event.clientY;
    ImgWidth = first_img.clientWidth;
    ImgHeight = first_img.clientHeight;
  let imgPosX = PosX - ImgPos[0];
  let imgPosY = PosY - ImgPos[1];
  let TrueimgPosX = Math.round((640/ImgWidth) * imgPosX);
  let TrueimgPosY = Math.round((480/ImgHeight) * imgPosY);
  const query = `${baseHost}/web_mouse_click?no=1&posX=${TrueimgPosX}&posY=${TrueimgPosY}`;

  fetch(query).then((response) => {
    console.log(
      `request to ${query} finished, status: ${response.status}`
    );
  });
}

function GetCoordinates2(event)
{
  let PosX = 0;
  let PosY = 0;
  let ImgWidth = 0;
  let ImgHeight = 0;
  let ImgPos;
  ImgPos = FindPosition(second_img)
  console.log(ImgPos);
    PosX = event.clientX;
    PosY = event.clientY;
    ImgWidth = second_img.clientWidth;
    ImgHeight = second_img.clientHeight;
  let imgPosX = PosX - ImgPos[0];
  let imgPosY = PosY - ImgPos[1];
  let TrueimgPosX = Math.round((640/ImgWidth) * imgPosX);
  let TrueimgPosY = Math.round((480/ImgHeight) * imgPosY);
  const query = `${baseHost}/web_mouse_click?no=2&posX=${TrueimgPosX}&posY=${TrueimgPosY}`;

  fetch(query).then((response) => {
    console.log(
      `request to ${query} finished, status: ${response.status}`
    );
  });
}
document.getElementById("first-cam-imgs").addEventListener("click", GetCoordinates);
document.getElementById("second-cam-imgs").addEventListener("click", GetCoordinates2);