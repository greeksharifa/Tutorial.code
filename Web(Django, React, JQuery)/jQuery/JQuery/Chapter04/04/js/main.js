$(function(){
    let duration=300;
    let $aside=$('.page-main > aside');
    let $asidButton = $aside.find('button')
        .on('click', function(){
            $aside.toggleClass('open');
            if($aside.hasClass('open')){
                $aside.stop(true).animate({
                    left: '-70px'
                }, duration, 'easeOutBack');
                $asidButton.find('img').attr('src', 'img/btn_close.png');
            } else{
                $aside.stop(true).animate({
                    left: '-350px'
                }, duration, 'easeInBack');
                $asidButton.find('img').attr('src', 'img/btn_open.png');
            }
        })
});