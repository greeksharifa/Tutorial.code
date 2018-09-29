$(function () {

    /*
     * 갤러리
     */
    $('#gallery').each(function () {

        // #gallery요소가 갤러리 컨테이너
        var $container = $(this);

        //옵션을 설정 Masonry를 준비
        $container.masonry({
            columnWidth: 230,
            gutter: 10,
            itemSelector: '.gallery-item'
        });

        // JSON 파일을 요청하고 성공하면 작업 실행
        $.getJSON('./data/content.json', function (data) {

            // 루프에서 생성 된 DOM 요소를 일시적으로 저장하는 배열
            var elements = [];

            // JSON 배열 (data)의 요소 (item)마다 반복
            $.each(data, function (i, item) {

                // 배열의 요소에서 HTML 문자열을 생성
                var itemHTML =
                        '<li class="gallery-item is-loading">' +
                            '<a href="' + item.images.large + '">' +
                                '<img src="' + item.images.thumb +
                                    '" alt="' + item.title + '">' +
                            '</a>' +
                        '</li>';

                // HTML 문자열을 DOM 요소 화하여 배열에 추가
                elements.push($(itemHTML).get(0));

            });

            // DOM을 삽입
            $container.append(elements);

            // 이미지로드가 완료되면 Masonry 레이아웃
            $container.imagesLoaded(function () {
                $(elements).removeClass('is-loading');
                $container.masonry('appended', elements);
            });
        });
    });
});
