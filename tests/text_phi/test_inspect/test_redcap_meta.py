"""Tests for scripts/text_phi/inspect/redcap_meta.py."""

from __future__ import annotations

import textwrap

import pytest

from scripts.text_phi.inspect.redcap_meta import (
    _checkbox_code_from_oid,
    _parse_checkbox_choices,
    parse_metadata,
)


def _minimal_xml() -> str:
    return textwrap.dedent("""\
        <?xml version="1.0" encoding="UTF-8" ?>
        <ODM xmlns="http://www.cdisc.org/ns/odm/v1.3"
             xmlns:redcap="https://projectredcap.org">
          <Study OID="P">
            <MetaDataVersion OID="V1">
              <FormDef OID="Form.f1" Name="Form One" redcap:FormName="f1">
                <ItemGroupRef ItemGroupOID="f1.grp"/>
              </FormDef>
              <ItemGroupDef OID="f1.grp" Name="grp">
                <ItemRef ItemOID="subject_number"/>
                <ItemRef ItemOID="subject_name"/>
                <ItemRef ItemOID="race___8"/>
              </ItemGroupDef>
              <ItemDef OID="subject_number" Name="subject_number"
                       DataType="integer" redcap:FieldType="text"
                       redcap:TextValidationType="int">
                <Question><TranslatedText>Subject Number</TranslatedText></Question>
              </ItemDef>
              <ItemDef OID="subject_name" Name="subject_name" DataType="text"
                       redcap:FieldType="text" redcap:Identifier="y">
                <Question><TranslatedText>Subject Name</TranslatedText></Question>
              </ItemDef>
              <ItemDef OID="race___8" Name="race___8" DataType="boolean"
                       redcap:Variable="race" redcap:FieldType="checkbox">
                <Question><TranslatedText>Race</TranslatedText></Question>
                <CodeListRef CodeListOID="race___8.choices"/>
              </ItemDef>
              <CodeList OID="race___8.choices" Name="race___8" DataType="boolean"
                        redcap:CheckboxChoices="8, Black | 9, Asian">
                <CodeListItem CodedValue="1"><Decode><TranslatedText>Checked</TranslatedText></Decode></CodeListItem>
                <CodeListItem CodedValue="0"><Decode><TranslatedText>Unchecked</TranslatedText></Decode></CodeListItem>
              </CodeList>
            </MetaDataVersion>
          </Study>
        </ODM>
    """)


def test_parses_minimal_xml(tmp_path):
    p = tmp_path / "m.xml"
    p.write_text(_minimal_xml())
    m = parse_metadata(p)
    assert m["counts"] == {"items": 3, "code_lists": 1, "forms": 1}


def test_item_carries_identifier_flag(tmp_path):
    p = tmp_path / "m.xml"
    p.write_text(_minimal_xml())
    m = parse_metadata(p)
    assert m["items"]["subject_name"]["identifier"] is True
    assert m["items"]["subject_number"]["identifier"] is False


def test_item_carries_data_type_and_validation(tmp_path):
    p = tmp_path / "m.xml"
    p.write_text(_minimal_xml())
    m = parse_metadata(p)
    assert m["items"]["subject_number"]["data_type"] == "integer"
    assert m["items"]["subject_number"]["validation_type"] == "int"


def test_item_carries_question(tmp_path):
    p = tmp_path / "m.xml"
    p.write_text(_minimal_xml())
    m = parse_metadata(p)
    assert m["items"]["subject_name"]["question"] == "Subject Name"


def test_checkbox_choice_label_resolved(tmp_path):
    p = tmp_path / "m.xml"
    p.write_text(_minimal_xml())
    m = parse_metadata(p)
    it = m["items"]["race___8"]
    assert it["field_type"] == "checkbox"
    assert it["checkbox_code"] == "8"
    assert it["checkbox_label"] == "Black"


def test_form_membership(tmp_path):
    p = tmp_path / "m.xml"
    p.write_text(_minimal_xml())
    m = parse_metadata(p)
    assert m["items"]["subject_name"]["form"] == "f1"
    assert m["forms"]["Form.f1"]["items"] == [
        "subject_number", "subject_name", "race___8"
    ]


def test_checkbox_code_extraction():
    assert _checkbox_code_from_oid("race___8") == "8"
    assert _checkbox_code_from_oid("path___21") == "21"
    assert _checkbox_code_from_oid("no_suffix") is None
    assert _checkbox_code_from_oid("race___abc_1") == "abc_1"


def test_checkbox_choices_parses_pipe_separated():
    got = _parse_checkbox_choices("8, Black | 9, Asian | 10, Caribbean")
    assert got == {"8": "Black", "9": "Asian", "10": "Caribbean"}


def test_checkbox_choices_preserves_empty_label():
    # REDCap uses `"1, "` for the DCC bad-session checkboxes; the empty
    # label must survive so column matching finds `(choice=)`.
    got = _parse_checkbox_choices("1, ")
    assert got == {"1": ""}


def test_checkbox_choices_empty_input():
    assert _parse_checkbox_choices(None) == {}
    assert _parse_checkbox_choices("") == {}


def test_missing_metadata_version_raises(tmp_path):
    p = tmp_path / "m.xml"
    p.write_text('<ODM xmlns="http://www.cdisc.org/ns/odm/v1.3"><Study/></ODM>')
    with pytest.raises(ValueError, match="MetaDataVersion"):
        parse_metadata(p)
