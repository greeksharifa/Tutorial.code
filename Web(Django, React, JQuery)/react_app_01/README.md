# react_app_01

## 환경 설정
/react_app_01

----/dev

--------`index.jsx`

----/output

--------`myCode.js`

----`index.html`

----`package.json`

----`webpack.config.js`

-------------------------------------------------------------------------------------

1. nodejs 설치
2. 터미널에서 `react_app_01` 폴더로 이동
3. `npm init`
4. 프로젝트 이름 입력(대문자 포함 불가) -> package.json 파일이 생성됨
5. `npm install react react-dom --save`
6. dev 폴더에 `index.jsx` 파일 생성
7. `npm install webpack --save`
8. `react_app_01` 폴더에 `webpack.config.js` 생성
9. `npm install babel-core babel-loader babel-preset-es2015 babel-preset-react --save`
10. package.json에 "dependencies" 다음에 추가:
```
"babel": {
  "presets": [
    "es2015",
    "reacf"
  ]
}
```
11. `webpack.config.js` 파일 수정
12. 빌드: `node_modules\.bin\webpack.cmd`

-------------------------------------------------------------------------------------

`index.html`
```
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
</head>
<body>
    <div id="container"></div>
    <script src="output/myCode.js"></script>
</body>
</html>
```

-------------------------------------------------------------------------------------

`index.jsx`
```
import React from "react";
import ReactDOM from "react-dom";

let HelloWorld = React.createClass({
        render: function(){
            return(
                <p> Hello {this.props.greetTarget}!</p>
            );
        }
    });
    ReactDOM.render(
        <div>
            <HelloWorld greetTarget={"Gorio"}/>
            <HelloWorld greetTarget={"World"}/>
            <HelloWorld greetTarget={"React"}/>
        </div>,
        document.querySelector("#container")
    );
```

-------------------------------------------------------------------------------------

`webpack.config.js`
```
let webpack = require("webpack");
let path = require("path");
var DEV = path.resolve(__dirname, "dev");
var OUTPUT = path.resolve(__dirname, "output");
var config = {
  entry: DEV + "/index.jsx",
  output: {
    path: OUTPUT,
    filename: "myCode.js"
  },
  module: {
    loaders: [{
      include: DEV,
      loader: "babel-loader",
    }]
  }
};

module.exports = config;
```
