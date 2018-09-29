$(function(){
    $('.slideshow').each(function(){
        let $slides = $(this).find('img'),
            slideCount = $slides.length,
            currentIndex = 0;
        $slides.eq(currentIndex).fadeIn();
        setInterval(showNextSlide, 2500);
        function showNextSlide(){
            let nextIndex = (currentIndex + 1) % slideCount;
            $slides.eq(currentIndex).fadeOut();
            $slides.eq(nextIndex).fadeIn();
            currentIndex = nextIndex;
        }
    })
});