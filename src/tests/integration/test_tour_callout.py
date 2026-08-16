from views.annomate.tour.callout import _TourCallout


class TestTourCalloutSizing:
    def test_long_body_text_does_not_overlap_counter_label(self, qtbot):
        """Verify a long, word-wrapped body string isn't clipped/overlapped.

        Regression test: QVBoxLayout's automatic heightForWidth pass doesn't
        reliably size a word-wrapped QLabel on the first adjustSize() call,
        which previously let the last line of a long step's body text
        overlap the step counter label below it. Success means the body
        label's bottom edge sits at or above the counter label's top edge
        after set_step() with realistic multi-line text.
        """
        callout = _TourCallout()
        qtbot.addWidget(callout)
        long_body = (
            "The File and Data menus above hold project save/open and "
            "export options. Replay this tour anytime from "
            "Help → Show Welcome Tour."
        )

        callout.set_step("You're all set", long_body, 10, 11, True, True)

        body_bottom = callout._body_lbl.geometry().bottom()
        counter_top = callout._counter_lbl.geometry().top()
        assert body_bottom <= counter_top

    def test_body_label_height_shrinks_for_short_text(self, qtbot):
        """Verify switching from a long step to a short one re-shrinks the label.

        Success means the body label's height for a one-line string is
        smaller than its height for a long, multi-line string.
        """
        callout = _TourCallout()
        qtbot.addWidget(callout)

        callout.set_step(
            "You're all set",
            "The File and Data menus above hold project save/open and "
            "export options. Replay this tour anytime from "
            "Help → Show Welcome Tour.",
            10,
            11,
            True,
            True,
        )
        long_height = callout._body_lbl.height()

        callout.set_step("Status Bar", "Short note.", 8, 11, True, False)
        short_height = callout._body_lbl.height()

        assert short_height < long_height
