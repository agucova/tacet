// Automatic tab selection based on language preference

import {
	getPreferredLanguage,
	LANGUAGE_CHANGE_EVENT,
	type Language,
} from './language-preference';

// Language label matching patterns
const LANGUAGE_LABELS: Record<Language, string[]> = {
	rust: ['Rust', 'rust', 'rs'],
	javascript: ['JavaScript', 'JS', 'TypeScript', 'TS', 'Node.js', 'Bun'],
	c: ['C'],
	cpp: ['C++', 'cpp'],
	go: ['Go', 'Golang'],
};

/**
 * Check if a tab label corresponds to a specific language.
 */
function getLanguageFromLabel(label: string): Language | null {
	for (const [lang, labels] of Object.entries(LANGUAGE_LABELS)) {
		if (labels.some((l) => label.includes(l))) {
			return lang as Language;
		}
	}
	return null;
}

/**
 * Check if a tab group contains language tabs.
 * A tab group is considered a language tab group if more than half of its tabs
 * are identified as language tabs.
 */
function isLanguageTabGroup(tabList: Element): boolean {
	const tabs = tabList.querySelectorAll('[role="tab"]');
	let languageTabCount = 0;

	tabs.forEach((tab) => {
		const label = tab.textContent?.trim() || '';
		if (getLanguageFromLabel(label)) {
			languageTabCount++;
		}
	});

	return languageTabCount > tabs.length / 2;
}

/**
 * Auto-select the tab matching the preferred language in a tab group.
 */
function autoSelectTab(tabList: Element, preferredLang: Language): boolean {
	const tabs = tabList.querySelectorAll('[role="tab"]');

	for (const tab of tabs) {
		const label = tab.textContent?.trim() || '';
		const tabLang = getLanguageFromLabel(label);

		if (tabLang === preferredLang && tab instanceof HTMLElement) {
			// Click the tab to select it
			tab.click();
			return true;
		}
	}

	return false;
}

/**
 * Process all tab groups on the page and auto-select language tabs.
 */
function processAllTabGroups(): void {
	const preferredLang = getPreferredLanguage();
	const tabLists = document.querySelectorAll('[role="tablist"]');

	tabLists.forEach((tabList) => {
		if (isLanguageTabGroup(tabList)) {
			autoSelectTab(tabList, preferredLang);
		}
	});
}

/**
 * Initialize auto-selection on page load.
 */
function init(): void {
	// Process tabs after DOM is fully loaded
	if (document.readyState === 'loading') {
		document.addEventListener('DOMContentLoaded', processAllTabGroups);
	} else {
		// DOM is already loaded
		processAllTabGroups();
	}

	// Listen for language preference changes
	window.addEventListener(LANGUAGE_CHANGE_EVENT, () => {
		processAllTabGroups();
	});

	// Re-process tabs after view transitions
	document.addEventListener('astro:page-load', () => {
		processAllTabGroups();
	});
}

// Run initialization
init();
