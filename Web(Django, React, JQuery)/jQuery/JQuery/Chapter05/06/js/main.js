$(function () {

    /*
     * Tabs
     */
    $('.work-section').each(function () {

        var $container = $(this),                            // a
            $navItems = $container.find('.tabs-nav li'),     // b
            $highlight = $container.find('.tabs-highlight'); // c
        // 탭의 각 요소를 jQuery 객체 화
        // a 탭과 패널을 포함한 전체 컨테이너
        // b 탭의 목록
        // c 선택한 탭의 하이라이트

        // jQuery UI Tabs를 실행
        $container.tabs({

            // 숨길 때의 애니메이션
            hide: { duration: 250 },

            // 표시 할 때 애니메이션
            show: { duration: 125 },

            // 로드시와 탭 선택시에 하이라이트의 위치를 조정
            create: moveHighlight,
            activate: moveHighlight
        });

        // 하이라이트의 위치를 조정하는 함수
        function moveHighlight (event, ui) {
            var $newTab = ui.newTab || ui.tab,  // 새로 선택된 탭의 jQuery 객체
                left = $newTab.position().left; // 새로 선택된 탭의 왼쪽 위치

            // 하이라이트의 위치를 애니메이션
            $highlight.animate({ left: left }, 500, 'easeOutExpo');
        }
    });

});
