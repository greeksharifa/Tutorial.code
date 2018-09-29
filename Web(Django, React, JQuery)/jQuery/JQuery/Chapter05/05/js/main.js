$(function () {

    /*
     * Tabs
     */
    $('#work').each(function () {

        // 탭의 각 요소를 jQuery 객체 화
        var $tabList    = $(this).find('.tabs-nav'),   // 탭의 목록
            $tabAnchors = $tabList.find('a'),          // 탭 (링크)
            $tabPanels  = $(this).find('.tabs-panel'); // 패널

        // 탭이 클릭되었을 때의 처리
        // 인자로 이벤트 객체를 받는다.
        $tabList.on('click', 'a', function (event) {

            // 링크 클릭에 대한 기본 동작을 취소
            event.preventDefault();

            // 클릭 된 탭을 jQuery 오브젝트화
            var $this = $(this);

            // 만약 이미 선택된 탭이라면 아무것도하지 않고 처리를 중지
            if ($this.hasClass('active')) {
                return;
            }

            // 모든 탭의 선택 상태를 해제 한,
            // 클릭 된 탭을 선택 상태로
            $tabAnchors.removeClass('active');
            $this.addClass('active');

            // 모든 패널을 일단 비 표시로하고
            // 클릭 된 탭에 대응하는 패널을 표시
            $tabPanels.hide();
            $($this.attr('href')).show();

        });

        // 첫 번째 탭을 선택 상태로
        $tabAnchors.eq(0).trigger('click');

    });

});
