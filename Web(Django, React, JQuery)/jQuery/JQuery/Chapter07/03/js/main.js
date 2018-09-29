$(function () {

    activateScene2();

    // Scene 2를 표시
    function activateScene2 () {

        var $content = $('#scene-2-content'),
            $charts = $content.find('.chart');

        // 컨텐츠가 오른쪽에서 나오는
        $content.stop(true).animate({
            right: 0
        }, 1200, 'easeInOutQuint');

        // 원형 차트 당 처리
        $charts.each(function(){
            var $chart = $(this),
                // "마스크"를 저장하고 각도 0으로 지정
                $circleLeft = $chart.find('.left .circle-mask-inner')
                    .css({ transform: 'rotate(0)' }),
                $circleRight = $chart.find('.right .circle-mask-inner')
                    .css({ transform: 'rotate(0)' }),
                // 백분율 값을 취득
                $percentNumber = $chart.find('.percent-number'),
                percentData = $percentNumber.text();

            // 백분율 표시를 일단 0
            $percentNumber.text(0);

            // 각도 애니메이션
            $({ percent: 0 }).delay(1000).animate({ percent: percentData }, {
                duration: 1500, 
                progress: function () {
                    var now = this.percent,
                        deg = now * 360 / 100,
                        degRight = Math.min(Math.max(deg, 0), 180),
                        degLeft  = Math.min(Math.max(deg - 180, 0), 180);
                    $circleRight.css({ transform: 'rotate(' + degRight + 'deg)' });
                    $circleLeft.css({ transform: 'rotate(' + degLeft + 'deg)' });
                    $percentNumber.text(Math.floor(now));
                }
            });
        });
    }

});
