from django import template

from pewtils import decode_text as pewtils_decode_text

register = template.Library()

@register.filter(name="decode_text")
def decode_text(value):
    return pewtils_decode_text(value)

@register.filter(name="unicode")
def to_unicode(value):
    return unicode(value)

@register.filter(name="values_list")
def values_list(queryset, key):
    return queryset.values_list(key, flat=True)
