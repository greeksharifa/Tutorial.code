$(function () {

    /*
     * Sticky header
     */
    $('.page-header').each(function () {

        var $window = $(window), // Window 객체
            $header = $(this),   // 헤더

            // 헤더의 복제
            $headerClone = $header.contents().clone(),

            // 헤더 복제 컨테이너
            $headerCloneContainer = $('<div class="page-header-clone"></div>'),

            // HTML의 위쪽에서 헤더의 저변까지의 거리 = 헤더의 최고 위치 + 헤더의 높이
            threshold = $header.offset().top + $header.outerHeight();

        // 컨테이너 헤더의 복제를 삽입
        $headerCloneContainer.append($headerClone);

        // 컨테이너를 body의 마지막에 삽입
        $headerCloneContainer.appendTo('body');

        // 스크롤시에 작업을 수행하지만, 횟수를 1 초당 15까지 제한
        $window.on('scroll', $.throttle(1000 / 15, function () {
            if ($window.scrollTop() > threshold) {
                $headerCloneContainer.addClass('visible');
            } else {
                $headerCloneContainer.removeClass('visible');
            }
        }));

        // 스크롤 이벤트를 발생시켜 초기 위치를 결정
        $window.trigger('scroll');
    });

});
