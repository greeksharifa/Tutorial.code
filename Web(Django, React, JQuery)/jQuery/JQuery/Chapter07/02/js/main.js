$(function () {

    initScene1();

    // Scene 1 : 이미지 시퀀스 애니메이션
    function initScene1 () {

        var
            // 이미지의 컨테이너와 그 안에있는 모든 이미지의 jQuery 객체
            $container       = $('#scene-1 .image-sequence'),
            $images          = $container.find('img'),

            // 이미지의 수와 현재 표시되는 이미지의 인덱스
            frameLength      = $images.length,
            currentFrame     = 0,

            // 애니메이션 작업에서 사용하는 수치
            counter          = 0, // 애니메이션의 진행 상황 카운터
            velocity         = 0, // 애니메이션의 속도

            // 애니메이션 타이머 (처음에는 빈)
            timer            = null,

            // 화면비율(width / height)
            imageAspectRatio = 864 / 486;

        // 컨테이너에서 마우스 휠 이벤트가 발생하면 작업 실행
        $container.on('mousewheel', function (event, delta) {
            // 마우스 휠의 방향에 따라 속도에 가산 또는 감산
            if (delta < 0) {
                velocity += 1.5;
            } else if (delta > 0) {
                velocity -= 1.5;
            }
            // 애니메이션을 시작하는 함수를 호출
            startAnimation();
        });

        // 애니메이션을 시작하는 함수
        function startAnimation () {
            // 이미 실행중인 애니메이션이 없으면 애니메이션을 실행
            if (!timer) {
                // 1/60 초마다 업데이트
                timer = setInterval(animateSequence, 1000 / 30);
            }
        }

        // 애니메이션을 종료하는 함수
        function stopAnimation () {
            clearInterval(timer);
            timer = null;
        }

        // 애니메이션의 함수
        function animateSequence () {

            // 새로 표시되는 이미지의 인덱스
            var nextFrame;

            // 속도에 마찰 계수를 걸어 호출 될 때마다 조금씩 줄여가는
            velocity *= 0.9;

            // 속도를 선택합니다. 0 ± 0.00001의 범위이면 0으로 간주 정지
            if (-0.00001 < velocity && velocity < 0.00001) {
                stopAnimation();
            } else {
                // 그 이외라면 카운터에 속도를 더할 때,
                // 카운터의 숫자를 이미지 수의 범위 내에서 제한
                counter = (counter + velocity) % frameLength;
            }

            // 카운터의 수치를 정수 화 해당 프레임을 표시
            nextFrame = Math.floor(counter);
            if (currentFrame !== nextFrame) {
                $images.eq(nextFrame).show();
                $images.eq(currentFrame).hide();
                currentFrame = nextFrame;
            }
        }

        // 컨테이너 종횡비를 유지하면서 표시 영역 가득 배치
        // 윈도우가 리사이즈 될 때마다 재조정된다
        $(window).on('resize', function () {

            // 윈도우의 폭과 높이를 취득
            var $window = $(this),
                windowWidth = $window.width(),
                windowHeight = $window.height();

            // 사진과 창의 가로 세로 비율을 비교하고
            // 컨테이너의 크기와 위치를 조정
            if (imageAspectRatio > windowWidth / windowHeight) {
                $container.css({
                    width: windowHeight * imageAspectRatio,
                    height: '100%',
                    top: 0,
                    left: (windowWidth - windowHeight * imageAspectRatio) / 2
                });
            } else {
                $container.css({
                    width: '100%',
                    height: windowWidth / imageAspectRatio,
                    top: (windowHeight - windowWidth / imageAspectRatio) / 2,
                    left: 0
                });
            }
        });

        // 창 크기 이벤트를 발생시킨 최초의 배치를 조정
        $(window).trigger('resize');
    }

});
