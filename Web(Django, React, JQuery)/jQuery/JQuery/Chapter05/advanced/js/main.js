$(function () {

    /*
     * Slideshow
     */
    $('.slideshow').each(function () {

    // 변수의 준비
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        var $container = $(this),                                 // a
            $slideGroup = $container.find('.slideshow-slides'),   // b
            $slides = $slideGroup.find('.slide'),                 // c
            $nav = $container.find('.slideshow-nav'),             // d
            $indicator = $container.find('.slideshow-indicator'), // e
            // 슬라이드 쇼의 각 요소의 jQuery 객체
            // a 슬라이드 쇼 전체 컨테이너
            // b 모든 슬라이드의 정리 (슬라이드 그룹)
            // c 각 슬라이드
            // d 네비게이션 (Prev/Next)
            // e 인디게이터 (ドット)

            slideCount = $slides.length, // 슬라이드 점수
            indicatorHTML = '',          // 인디게이터 콘텐트
            currentIndex = 0,            // 현재 슬라이드의 인덱스
            duration = 500,              // 다음 슬라이드에 애니메이션의 소요 시간
            easing = 'easeInOutExpo',    // 다음 슬라이드에 애니메이션의 여유 종류
            interval = 7500,             // 자동으로 다음 슬라이드로 옮길 때까지의 시간
            timer;                       // 타이머


    // HTML 요소의 배치 생성 삽입
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        // 각 슬라이드의 위치를 결정하고,
        // 해당 표시기의 앵커를 생성
        $slides.each(function (i) {
            $(this).css({ left: 100 * i + '%' });
            indicatorHTML += '<a href="#">' + (i + 1) + '</a>';
        });

        // 인디게이터에 컨텐츠를 삽입
        $indicator.html(indicatorHTML);


    // 함수의 정의
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        // 모든 슬라이드를 표시하는 함수
        function goToSlide (index) {
            // 슬라이드 그룹을 대상 위치에 맞게 이동
            $slideGroup.animate({ left: - 100 * index + '%' }, duration, easing);
            // 현재 슬라이드의 인덱스를 덮어쓰기
            currentIndex = index;
            // 탐색 및 표시 상태를 업데이트
            updateNav();
        }

        // 슬라이드의 상태에 따라 탐색 및 표시를 업데이트하는 함수
        function updateNav () {
            var $navPrev = $nav.find('.prev'), // Prev (뒤로) 링크
                $navNext = $nav.find('.next'); // Next (앞으로) 링크
            // 만약 첫 번째 슬라이드이라면 Prev 탐색을 해제
            if (currentIndex === 0) {
                $navPrev.addClass('disabled');
            } else {
                $navPrev.removeClass('disabled');
            }
            // 만약 마지막 슬라이드이라면 Next 탐색을 해제
            if (currentIndex === slideCount - 1) {
                $navNext.addClass('disabled');
            } else {
                $navNext.removeClass('disabled');
            }
            // 현재 슬라이드의 표시를 해제
            $indicator.find('a').removeClass('active')
                .eq(currentIndex).addClass('active');
        }

        // 타이머를 시작하는 함수
        function startTimer () {
            // 변수 interval에서 설정 한 시간이 경과 할 때마다 작업을 수행
            timer = setInterval(function () {
                // 현재 슬라이드의 인덱스에 따라 다음 표시 할 슬라이드의 결정
                // 만약 마지막 슬라이드이라면 첫 번째 슬라이드에
                var nextIndex = (currentIndex + 1) % slideCount;
                goToSlide(nextIndex);
            }, interval);
        }

        // 타이머를 중지있는 함수
        function stopTimer () {
            clearInterval(timer);
        }


    // 인벤토리 등록
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        // 네비게이션 링크를 클릭하면 해당 슬라이드를 표시
        $nav.on('click', 'a', function (event) {
            event.preventDefault();
            if ($(this).hasClass('prev')) {
                goToSlide(currentIndex - 1);
            } else {
                goToSlide(currentIndex + 1);
            }
        });

        // 인디게이터의 링크를 클릭하면 해당 슬라이드를 표시
        $indicator.on('click', 'a', function (event) {
            event.preventDefault();
            if (!$(this).hasClass('active')) {
                goToSlide($(this).index());
            }
        });

        // 마우스오버면 타이머를 정지 그렇지 않으면 시작
        $container.on({
            mouseenter: stopTimer,
            mouseleave: startTimer
        });


    // 슬라이드 쇼 시작
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        // 첫 번째 슬라이드를 표시
        goToSlide(currentIndex);

        // 타이머를 시작
        startTimer();

    });

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

        // 컨테이너를 body에 삽입
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

            // 볼 때 애니메이션
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

    // Google Maps
    function initMap () {
        var mapContainer = document.getElementById('map-container'),
            mapImageSrc = mapContainer.getElementsByTagName('img')[0].getAttribute('src'),
            mapParams = decodeURIComponent(mapImageSrc.split('?')[1]).split('&'),
            mapData = {},
            mapStyleName = 'Mono',
            mapStyles = [
                {
                    featureType: 'all',
                    elementType: 'all',
                    stylers: [
                        { visibility: 'on' },
                        { hue: '#105ea7' },
                        { saturation: -100 },
                        { invert_lightness: true }
                    ]
                },
                {
                    elementType: 'labels.icon',
                    stylers: [
                        { visibility: 'off' }
                    ]
                }
            ],
            latLng,
            mapOptions,
            map,
            marker,
            markerLatLng,
            i,
            len,
            pair;
        for (i = 0, len = mapParams.length; i < len; i++) {
            pair = mapParams[i].split('=');
            mapData[pair[0]] = pair[1];
        }
        markerLatLng = mapData.markers? mapData.markers.split(','): null;
        latLng = mapData.center? mapData.center.split(','): markerLatLng;
        mapOptions = {
            center: new google.maps.LatLng(latLng[0], latLng[1]),
            disableDefaultUI: true,
            panControl: true,
            zoom: +mapData.zoom || 16,
            zoomControl: true,
            zoomControlOptions: {
                style: google.maps.ZoomControlStyle.SMALL
            }
        };
        map = new google.maps.Map(mapContainer, mapOptions);
        map.mapTypes.set(mapStyleName, new google.maps.StyledMapType(mapStyles, { name: mapStyleName }));
        map.setMapTypeId(mapStyleName);
        if (mapData.markers) {
            marker = new google.maps.Marker({
                position: new google.maps.LatLng(markerLatLng[0], markerLatLng[1]),
                map: map
            });
        }        
    }

    initMap();

});
