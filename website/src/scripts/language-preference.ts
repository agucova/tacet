// Language preference management with localStorage persistence

export type Language = 'rust' | 'javascript' | 'c' | 'cpp' | 'go';

const LANGUAGE_KEY = 'tacet-preferred-language';
export const LANGUAGE_CHANGE_EVENT = 'tacet:language-change';

/**
 * Get the user's preferred language from localStorage.
 * Defaults to 'rust' if no preference is set.
 */
export function getPreferredLanguage(): Language {
	if (typeof window === 'undefined') return 'rust';
	const stored = localStorage.getItem(LANGUAGE_KEY);
	if (stored && isValidLanguage(stored)) {
		return stored as Language;
	}
	return 'rust';
}

/**
 * Set the user's preferred language and persist to localStorage.
 * Emits a custom event that other parts of the app can listen to.
 */
export function setPreferredLanguage(lang: Language): void {
	if (typeof window === 'undefined') return;
	localStorage.setItem(LANGUAGE_KEY, lang);
	window.dispatchEvent(
		new CustomEvent(LANGUAGE_CHANGE_EVENT, {
			detail: { language: lang },
		})
	);
}

/**
 * Check if a string is a valid Language type.
 */
function isValidLanguage(lang: string): lang is Language {
	return ['rust', 'javascript', 'c', 'cpp', 'go'].includes(lang);
}
