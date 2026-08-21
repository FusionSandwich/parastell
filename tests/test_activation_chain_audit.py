from parastell.activation.chain_audit import audit_activation_chain


def _files(tmp_path):
    chain = tmp_path / "chain.xml"
    chain.write_text(
        """<depletion_chain>
<nuclide name="Cu63"><reaction type="(n,gamma)" target="Cu64" /></nuclide>
<nuclide name="Cu64" half_life="1"><decay type="beta-" target="Zn64" />
<source particle="photon" /></nuclide>
<nuclide name="Zn64" />
</depletion_chain>""",
        encoding="ascii",
    )
    cross_sections = tmp_path / "cross_sections.xml"
    cross_sections.write_text(
        '<cross_sections><library materials="Cu63 Cu64" path="cu.h5" /></cross_sections>',
        encoding="ascii",
    )
    return chain, cross_sections


def test_chain_audit_reports_release_mismatch(tmp_path):
    chain, cross_sections = _files(tmp_path)
    report = audit_activation_chain(
        chain,
        cross_sections,
        ["Cu63"],
        chain_release="ENDF/B-VIII.0",
        transport_release="ENDF/B-VII.1",
    )
    assert not report.passes
    assert report.release_mismatches


def test_chain_audit_passes_matching_stable_nuclide(tmp_path):
    chain, cross_sections = _files(tmp_path)
    report = audit_activation_chain(
        chain,
        cross_sections,
        ["Cu-63"],
        chain_release="TENDL-2023",
        transport_release="TENDL-2023",
    )
    assert report.passes
    assert not report.missing_transport
    assert report.reachable_chain_nuclides == ("Cu63", "Cu64", "Zn64")
    assert report.photon_source_nuclides == ("Cu64",)


def test_chain_audit_rejects_missing_daughter(tmp_path):
    chain, cross_sections = _files(tmp_path)
    chain.write_text(
        '<depletion_chain><nuclide name="Cu63">'
        '<reaction type="(n,gamma)" target="Missing64" />'
        "</nuclide></depletion_chain>",
        encoding="ascii",
    )
    report = audit_activation_chain(chain, cross_sections, ["Cu63"])
    assert not report.passes
    assert report.missing_chain_targets == ("Cu63->Missing64",)
