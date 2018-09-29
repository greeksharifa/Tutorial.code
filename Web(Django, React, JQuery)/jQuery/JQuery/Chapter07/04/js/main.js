$(function () {

    initScene3();

    // Scene 3 : 마스크 애니메이션
    function initScene3 () {

        var $container = $('#scene-3'),            // 컨테이너
            $masks     = $container.find('.mask'), // 마스크
            $lines     = $masks.find('.line'),     // 경계
            maskLength = $masks.length,            // 마스크의 총수

            // 각 마스크 자르기 영역의 데이터를 저장할 수있는 배열
            maskData   = [];

        // 마스크마다 자르기 영역의 왼쪽 좌표를 저장
        $masks.each(function (i) {
            maskData[i] = { left: 0 };
        });

        // 마스크에 마우스오버 했을 때와 잃을 때 작업을 수행
        $container.on({
            mouseenter: function () {
                resizeMask($(this).index());
            },
            mouseleave: function () {
                resizeMask(-1);
            }
        }, '.mask');

        // 각 마스크의 초기 자르기 영역과 경계의 위치를 지정
        resizeMask(-1);

        // 각 마스크 자르기 영역과 경계의 위치를 애니메이션시키는 함수
        function resizeMask (active) {

            // 컨테이너의 폭과 높이를 취득하고
            // 각 자르기 영역의 오른쪽과 아래쪽의 좌표로
            var w = $container.width(),
                h = $container.height();

            // 마스크마다 처리
            $masks.each(function (i) {

                var $this = $(this), // 이 마스크
                    l;               // 자르기 영역의 왼쪽 좌표

                // active = 마우스오버된 마스크의 인덱스
                //          -1이면 마우스가 빠진 상태
                // i      = 이 마스크의 인덱스

                // 마우스 이벤트에 의해 마스크 노 자르기 영역의 왼쪽 좌표를 산출
                if (active === -1) {
                    // 마우스를 벗어난 경우 균등하게 할당
                    l = w / maskLength * i;
                } else if (active < i) {
                    // 마우스오버된 마스크보다 오른쪽 마스크는
                    // 자르기 영역의 왼쪽이 오른쪽 방향으로 수정된다
                    l = w * (1 - 0.1 * (maskLength - i));
                } else {
                    // 그 외에는 왼쪽이 왼쪽으로
                    l = w * 0.05 * i;
                }

                // maskData [i]에 저장되어있는 왼쪽의 좌표를
                // l의 수치까지 애니메이션시키는
                $(maskData[i]).stop(true).animate({ left: l }, {
                    duration: 1000,
                    easing: 'easeOutQuart',
                    // 마스크와 경계의 CSS를 다시 작성
                    progress: function () {
                        var now = this.left;
                        $this.css({
                            // 각 숫자를 rect () 형식으로 건설
                            clip: 'rect(0px ' + w + 'px ' +
                                    h + 'px ' + now + 'px)'
                        });
                        $this.find($lines).css({
                            left: now
                        });
                    }
                });
            });
        }
    }

});
