$(function () {

    /*
     * Back-toTop button (Smooth scroll)
     */
    $('.back-to-top').on('click', function () {

        // Smooth Scroll 플러그인을 실행
        $.smoothScroll({
            easing: 'easeOutExpo', // 이징의 종류
            speed: 500             // 소요 시간
        });
    });

});
