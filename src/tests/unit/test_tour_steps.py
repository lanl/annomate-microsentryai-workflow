from views.annomate.tour.steps import build_default_steps


class TestBuildDefaultSteps:
    def test_returns_seventeen_steps_in_order(self):
        """Verify build_default_steps() returns the documented 17-step tour in order.

        Success means the step keys match the exact sequence described in the
        feature plan, from the welcome card through the outro card. The
        "annotations"/"inspector" steps were folded into "navigator" once
        those sections moved inline into the dataset navigator's cards;
        "project_start", "active_tool", "view_overlays", and
        "image_adjustments" were added to cover UI surface introduced by the
        dataset-navigator/activity-bar redesign; and "navigator_markers",
        "center_crop", "grid", and "anomaly_constraints" were added as
        dedicated deep-dive stops instead of folding that detail into their
        parent steps' body text.
        """
        steps = build_default_steps()
        assert [s.key for s in steps] == [
            "welcome",
            "project_start",
            "canvas",
            "tool_palette",
            "viewport_actions",
            "navigator",
            "navigator_markers",
            "active_tool",
            "classes",
            "microsentry",
            "view_overlays",
            "center_crop",
            "grid",
            "anomaly_constraints",
            "image_adjustments",
            "status_bar",
            "outro",
        ]

    def test_keys_are_unique(self):
        """Verify no two steps share the same key.

        Success means the set of keys has the same length as the step list.
        """
        steps = build_default_steps()
        keys = [s.key for s in steps]
        assert len(keys) == len(set(keys))

    def test_titles_and_bodies_are_non_empty(self):
        """Verify every step has non-empty title/body text to display.

        Success means every step's title and body are non-empty strings.
        """
        for step in build_default_steps():
            assert step.title.strip()
            assert step.body.strip()

    def test_only_welcome_and_outro_have_no_target(self):
        """Verify only the welcome/outro steps are centered cards with no spotlight.

        All other steps must provide a target resolver so the overlay can
        highlight a real widget. Success means exactly {"welcome", "outro"}
        have target=None and every other step has a callable target.
        """
        steps = build_default_steps()
        no_target_keys = {s.key for s in steps if s.target is None}
        assert no_target_keys == {"welcome", "outro"}
        for step in steps:
            if step.key not in no_target_keys:
                assert callable(step.target)
