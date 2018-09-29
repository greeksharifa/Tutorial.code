$(function () {

    /*
     * Back-toTop button (Smooth scroll)
     */
    $('.back-to-top').each(function () {

        // html 또는 body 중 스크롤 가능한 요소를 감지
        var $el = $(scrollableElement('html', 'body'));

        // 버튼에 클릭 이벤트를 설정
        $(this).on('click', function (event) {
            event.preventDefault();
            $el.animate({ scrollTop: 0 }, 2750);
        });
    });

    // scrollTop을 사용할 수있는 요소를 감지하는 함수
    // http://www.learningjquery.com/2007/10/improved-animated-scrolling-script-for-same-page-links#update4
    function scrollableElement (elements) {
        var i, len, el, $el, scrollable;
        for (i = 0, len = arguments.length; i < len; i++) {
            el = arguments[i],
            $el = $(el);
            if ($el.scrollTop() > 0) {
                return el;
            } else {
                $el.scrollTop(1);
                scrollable = $el.scrollTop() > 0;
                $el.scrollTop(0);
                if (scrollable) {
                    return el;
                }
            }
        }
        return [];
    }

});
