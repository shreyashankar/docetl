import React, { useState } from 'react';
import { Input } from '@/components/ui/input';
import { useBookmarkContext } from '@/contexts/BookmarkContext';
import { Bookmark, ChevronDown, ChevronUp, Filter, Trash2 } from 'lucide-react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"

const BookmarksPanel: React.FC = () => {
  const { bookmarks, removeBookmark } = useBookmarkContext();
  const [searchTerm, setSearchTerm] = useState('');
  const [expandedBookmarkId, setExpandedBookmarkId] = useState<string | null>(null);
  const [selectedColor, setSelectedColor] = useState<string | 'all'>('all');

  const uniqueColors = Array.from(new Set(bookmarks.map(bookmark => bookmark.color)));

  const filteredBookmarks = bookmarks.filter(bookmark =>
    bookmark.text.toLowerCase().includes(searchTerm.toLowerCase()) &&
    (selectedColor === 'all' || bookmark.color === selectedColor)
  );

  const toggleBookmarkExpansion = (id: string) => {
    setExpandedBookmarkId(expandedBookmarkId === id ? null : id);
  };

  const handleDeleteBookmark = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    removeBookmark(id);
    if (expandedBookmarkId === id) {
      setExpandedBookmarkId(null);
    }
  };

  return (
    <div className="h-full p-4 flex flex-col">
      <h2 className="text-sm font-bold mb-2 flex items-center uppercase">
        <Bookmark className="mr-2" size={16} />
        Bookmarks
      </h2>
      <div className="flex mb-2">
        <Input
          type="text"
          placeholder="Search..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="flex-grow mr-2"
        />
        <Popover>
          <PopoverTrigger asChild>
            <Button variant="ghost" size="icon" className="relative">
              <Filter className="h-4 w-4" />
              {selectedColor !== 'all' && (
                <span className="absolute -top-2 -right-2 bg-primary text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
                  1
                </span>
              )}
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-48">
            <Select value={selectedColor} onValueChange={(value) => setSelectedColor(value as string)}>
              <SelectTrigger className="border-none shadow-none">
                <SelectValue placeholder="Filter by color" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All colors</SelectItem>
                {uniqueColors.map((color) => (
                  <SelectItem key={color} value={color}>
                    <div className="flex items-center">
                      <div
                        className="w-3 h-3 rounded-full mr-2"
                        style={{ backgroundColor: color }}
                      />
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </PopoverContent>
        </Popover>
      </div>
      <div className="overflow-y-auto flex-grow">
        {filteredBookmarks.map(bookmark => (
          <div key={bookmark.id} className="mb-2">
            <div 
              className="flex items-center cursor-pointer"
              onClick={() => toggleBookmarkExpansion(bookmark.id)}
            >
              <div
                className="w-3 h-3 rounded-full mr-2"
                style={{ 
                  backgroundColor: bookmark.color,
                  minWidth: '0.75rem',
                  minHeight: '0.75rem'
                }}
              />
              <span className="whitespace-nowrap overflow-hidden text-ellipsis">
                {bookmark.text}
              </span>
              {expandedBookmarkId === bookmark.id ? (
                <ChevronUp className="ml-auto min-w-4 min-h-4" size={16} />
              ) : (
                <ChevronDown className="ml-auto min-w-4 min-h-4" size={16} />
              )}
            </div>
            {expandedBookmarkId === bookmark.id && (
              <div className="mt-2 ml-5 text-sm text-gray-600">
                <p className="whitespace-nowrap overflow-hidden text-ellipsis">Source: {bookmark.source}</p>
                {bookmark.notes.map((note, index) => (
                  <p key={index} className="whitespace-nowrap overflow-hidden text-ellipsis">Note {index + 1}: {note.note}</p>
                ))}
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={(e) => handleDeleteBookmark(e, bookmark.id)}
                  className="mt-2"
                >
                  <Trash2 size={16} className="mr-2" />
                  Delete
                </Button>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default BookmarksPanel;