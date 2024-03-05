const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// 478개의 포인트 중 추출해야할 68개의 인덱스
const landmark_68_index = [0, 5, 6, 14, 17, 20, 33, 37, 40, 61,
                           63, 66, 70, 75, 82, 84, 88, 91, 94, 105,
                           107, 136, 137, 144, 150, 152, 153, 158, 160, 172,
                           173, 176, 177, 195, 197, 215, 227, 263, 267, 269,
                           270, 290, 293, 296, 300, 308, 310, 312, 314, 321,
                           334, 336, 365, 366, 373, 377, 378, 380, 385, 387,
                           397, 398, 401, 402, 405, 435, 447, 455];


// 68개 랜드마크 포인트 저장할 배열
var landmark_68 = new Array();
// 소요시간 측정을 위한 변수
var start = 0;
var end = 0;
var duration = 0;
// 몰입도 측정 결과 변수
var immResult = '얼굴인식불가';
// 몰입도 측정 결과 리스트
let resultArr = new Array();
let resultTime = new Array();

// 랜드마크 포인트 추출 모델 로드
const faceMesh = new FaceMesh({locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
}});
// 랜드마크 포인트 추출 모델 사용을 위한 설정값 세팅
faceMesh.setOptions({
    maxNumFaces: 1,
    refineLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});
// callback 함수 호출
// 모델의 결과가 존재할 때 호출
faceMesh.onResults(onResults);

// callback 함수
// 478개의 랜드마크 포인트 중 68개만 추출하여 캔버스에 드로우
function onResults(results) {
    start = window.performance.now();
    landmark_68.length = 0;
    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(
        results.image, 0, 0, canvas.width, canvas.height);

    if (results.multiFaceLandmarks) {
        for (const landmarks of results.multiFaceLandmarks) {
            for (var index of landmark_68_index) {
                landmark_68.push(landmarks[index]);
            }
            drawLandmarks(ctx, landmark_68, {color: '#30FF30', radius: 1});
        }
        predict_immersion();
    }
    ctx.restore();
}


// 실시간 웹캠 데이터 취득
// 비디오 태그를 모델의 인풋으로 전달
const camera = new Camera(video, {
    onFrame: async () => {
        await faceMesh.send({image: video});
    },
    width: 224,
    height: 224
});
camera.start();

// 두 포인트의 거리 계산 함수
function cal_distance(p1, p2){
    var x1 = p1.x;
    var y1 = p1.y;
    var x2 = p2.x;
    var y2 = p2.y;

    return Math.sqrt(
        Math.pow((x2 - x1), 2) +
        Math.pow((y2 - y1), 2)
        );
}

// 몰입도 진단 모델 사용 함수
let model;
let first = true;
// async function loadModel() {
//     return await tf.loadLayersModel("https://loca.aido.services/model_fm/model.json");
// }
// model = loadModel();
async function predict_immersion() {
    try {
        // 몰입도 진단 모델 호출
        // 첫 실행시에 호출되고 이후는 브라우저 캐시에 저장됨
        if (first) {
            model = await tf.loadLayersModel("https://loca.aido.services/model_fm/model.json");
            if (model == null){
                return
            }
            first = false;
        }

        const distance = new Array();
        
        for (let i = 0; i < landmark_68.length; i++) {
            for (let j = i + 1; j < landmark_68.length; j++) {
                distance.push(cal_distance(landmark_68[i], landmark_68[j]));
            }
        }
        
        if (distance.length != 2278) {
            // memory leak 이슈로 삭제
            // tf.dispose(model);
            // first = true;
            immResult = '얼굴인식불가';
            throw new Error('랜드마크 인식 안됨');
        }
        // console.log(distance.length);
        // console.log(model);
        // 이미지 텐서화, 모델을 사용하려면 텐서화를 거쳐야됨
        var input = tf.tensor1d(distance);
        var input_reshape = input.reshape([1,2278]);
        const pred = model.predict(input_reshape);
        const pred_arr1 = pred[0].dataSync();

        var result;
        if (pred_arr1 < 0.5) {
            result = '집중';
        } else {
            result = '집중X';
        }

        // 모델 사용, immResult에 몰입도 진단 결과와 확률값 저장
        immResult = tf.tidy(() => {
            const pred = model.predict(input.reshape([1,2278]));
            const pred_arr1 = pred[0].dataSync();

            var result;
            if (pred_arr1 < 0.5) {
                result = '집중';
            } else {
                result = '집중X';
            }
            return result;
        });
        // 텐서 release -> 텐서를 사용한 다음에는 꼭 release를 해줘야함
        // 자바스크립트에서 자동으로 해주지 않
    } catch (err) {
        console.log(err.message);
    } finally {
        errMessage = 0;
        end = window.performance.now();
        duration = end - start;
        tf.dispose(input_reshape);
        tf.dispose(input);
        distance = null;
        // console.log('소요시간(ms):', duration);
    }
}


// correct 배열의 true-false 비율을 계산하여 양불판정
async function final_result() {
    // 저장 종료 타임 스탬프
    let saveEndTime = window.performance.now();

    // 로그 타임스탬프 생성
    var today = new Date();
    var year = today.getFullYear();
    var month = ('0' + (today.getMonth() + 1)).slice(-2);
    var day = ('0' + today.getDate()).slice(-2);
    var dateString = year + '-' + month  + '-' + day;
    var hours = ('0' + today.getHours()).slice(-2); 
    var minutes = ('0' + today.getMinutes()).slice(-2);
    var seconds = ('0' + today.getSeconds()).slice(-2); 
    var timeString = hours + ':' + minutes  + ':' + seconds;
    var insert_dt = dateString + ' ' + timeString;

    // const label = document.getElementById("imm");
    // const real = label.textContent;
    // document.getElementById('imm').innerHTML = insert_dt + ' '+ immResult + '\n' + real;
    document.getElementById('imm').innerHTML = '정상 동작 여부:' + ' ' + (immResult == '얼굴인식불가' ? 'X' : 'O');

    
    let sum = resultTime.reduce((accumulator, currentValue) => accumulator + currentValue, 0);
    let average = sum / resultTime.length;

    // 저장 시작부터 저장 종료까지의 시간 설정
    if (saveEndTime - saveStartTime >= 100*1000) {
        if (saveFlag) {
            content_result = JSON.stringify({
                'loss': (100 - resultArr.length) / 100,
                'immersion': resultArr,
                'avg_time': average,
                'times': resultTime
            });
        
            // 로컬에 json 저장
            var blob = new Blob([content_result], { type: 'application/json' });
            var a = document.createElement('a');
            a.href = window.URL.createObjectURL(blob);
            a.download = 'result.json';
            a.click();
        }
        saveFlag = false;
        resultArr.length = 0;
        resultTime.length = 0;
    }

    if (saveFlag) {
        console.log(saveEndTime - saveStartTime);
        resultArr.push(immResult);
        resultTime.push(duration);
    }
}

// 9초마다 양불판정 함수 호출
setInterval(final_result, 1000);


function resetImmLog() {
    document.getElementById('imm').innerHTML = '';
}


let saveFlag = false;
let saveStartTime;
function startTest() {
    saveFlag = true;
    resetImmLog();
    saveStartTime = window.performance.now();
}



function next_page() {
    var video = document.getElementById("content");
    var source = video.getElementsByTagName("source")[0]; // 첫 번째 <source> 요소
    var videoPath = source.getAttribute("src");

    if (videoPath!='/static/기본/평균구하기_기본.mp4') {
        fetch("/next-page", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({video_path: videoPath})
        })
        .then(response => response.json())
        .then(data => {
            if (data.new_video_id) {
                window.location.href = `/${data.new_video_id}`;
            }
        })
        .catch(error => console.error('Error:', error));   
    } else {
        alert('마지막 영상입니다. 추천 영상을 시청해주세요.');
    }
}
    
// function show_reco() {
//     fetch("/recomendation", {
//         method: "POST",
//         headers: {
//             "Content-Type": "application/json",
//         },
//         body: JSON.stringify({video_path: videoPath})
//     })
//     .then(response => response.json())
//     .catch(error => console.error('Error:', error));
// }




var myVideo = document.getElementById('content');
var intervalId = null;
var tot_imm = 0;
var cnt_imm = 0;

function count_immersion() {
    tot_imm += 1;
    cnt_imm += (immResult == '집중' ? 1 : 0);
}

myVideo.addEventListener('play', function() {
    // 동영상 재생 시작 시 실행할 코드
    if (intervalId) {
        clearInterval(intervalId);
    }
    var button = document.getElementById("next");
    button.disabled = true;
    
    var button = document.getElementById("reco");
    button.disabled = true;

    intervalId = setInterval(count_immersion, 10); // 매 초마다 메시지 출력
});

myVideo.addEventListener('ended', function() {
    // 동영상 재생 종료 시 실행할 코드
    clearInterval(intervalId);
    intervalId = null;
    console.log('동영상 재생 종료');

    var video = document.getElementById("content");
    var source = video.getElementsByTagName("source")[0]; // 첫 번째 <source> 요소
    var videoPath = source.getAttribute("src");
    
    fetch("/video_ended", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({video_path: videoPath, imm_count: cnt_imm/tot_imm})
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
    })
    .catch(error => console.error('Error:', error));
    
    if (videoPath=='/static/기본/평균구하기_기본.mp4') {
        alert('추천 영상을 시청해주세요.');
    } else {
        alert('다음 영상을 시청해주세요.');
    }

    var button = document.getElementById("next");
    button.disabled = false;
    
    var button = document.getElementById("reco");
    button.disabled = false;
});
