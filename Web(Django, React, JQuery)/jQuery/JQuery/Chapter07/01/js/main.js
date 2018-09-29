$(function () {

    // 진행률 표시 함수를 호출
    imagesProgress();

    // 진행률 표시 함수를 호출
    function imagesProgress () {

        var $container    = $('#progress'),                    // 1
            $progressBar  = $container.find('.progress-bar'),  // 2
            $progressText = $container.find('.progress-text'), // 3
            // 1. 진행률 표시 전체 컨테이너
            // 2. 진행률 표시 막대 부분
            // 3. 진행률 표시 텍스트 부분

            // imagesLoaded 라이브러리에서 body 요소의 이미지 로딩을 모니터링
            // 동시에 body 전체 이미지 수를 저장
            imgLoad       = imagesLoaded('body'),
            imgTotal      = imgLoad.images.length,

            // 읽을 완료 한 이미지의 숫자 카운터와
            // 진행률 표시의 현재 위치에 해당하는 수치 (모두 처음에는 0)
            imgLoaded     = 0,
            current       = 0,

            // 1 초에 60 번씩 읽어 여부 확인
            progressTimer = setInterval(updateProgress, 1000 / 60);

        // imagesLoaded을 이용하여 이미지를로드 할 때마다 카운터를 가산
        imgLoad.on('progress', function () {
            imgLoaded++;
        });

        // 이미지로드 상황을 바탕으로 진행 표시를 업데이트
        // 이 함수는 setInterval () 메소드에 의해 1 초에 60 번 불려
        function updateProgress () {

            // 읽을 완료 한 이미지의 비율
            var target = (imgLoaded / imgTotal) * 100;

            // current (현재 위치)와 target (목적지)의 거리를 바탕으로 여유를 건다
            current += (target - current) * 0.1;

            // 표시 바의 폭과 텍스트에 current 값을 반영
            // 텍스트는 소수점 이하를 버리고 정수로
            $progressBar.css({ width: current + '%' });
            $progressText.text(Math.floor(current) + '%');

            // 종료
            if(current >= 100){
                // 진행률 표시의 업데이트를 중지
                clearInterval(progressTimer);
                // CSS 스타일을 바꾸기 위하여 클래스를 추가
                $container.addClass('progress-complete');
                // 진행률 막대와 텍스트를 동시에 애니메이션시키기 위해
                // 그룹화하고 하나의 jQuery 객체에
                $progressBar.add($progressText)
                    // 0.5 초 대기
                    .delay(500)
                    // 0.25 초에 걸쳐 진행률 막대와 텍스트를 투명하게
                    .animate({ opacity: 0 }, 250, function () {
                        // 1 초에 걸쳐 오버레이를 위쪽으로 슬라이드 아웃
                        $container.animate({ top: '-100%' }, 1000, 'easeInOutQuint');
                    });
            }

            // current가 99.9보다 크면 100으로 간주하여 종료
            if (current > 99.9) {
                current = 100;
            }
        }
    }

});
